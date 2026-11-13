#include "server.h"

#include "forward.h"
#include "json.h"
#include "metal.h"
#include "model.h"
#include "sampler.h"
#include "tokenizer.h"

#include <arpa/inet.h>
#include <errno.h>
#include <netdb.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <time.h>
#include <unistd.h>

#define DSM_MAX_BODY (8U * 1024U * 1024U)
#define DSM_MAX_OUTPUT (8U * 1024U * 1024U)
#define DSM_DEFAULT_CONTEXT 32768U
#define LIT(text) text, sizeof(text) - 1

typedef struct {
  dsm_model model;
  dsm_tokenizer *tokenizer;
  dsm_forward *forward;
  dsm_metal *metal;
  float *logits;
  int32_t *cached;
  size_t cached_len, cached_cap;
  char model_id[256];
  uint32_t context;
  uint32_t threads;
  int use_metal;
} server;
typedef struct {
  char *data;
  size_t len, cap;
} string;

static int append_assistant_content(const dsm_json *value, string *out);

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + t.tv_nsec / 1e9;
}
static int send_all(int fd, const void *buf, size_t len) {
  const char *p = buf;
  while (len) {
    ssize_t n = send(fd, p, len, 0);
    if (n <= 0)
      return 0;
    p += n;
    len -= (size_t)n;
  }
  return 1;
}
static int append(string *s, const char *p, size_t n) {
  if (s->len + n + 1 > s->cap) {
    size_t cap = s->cap ? s->cap : 256;
    while (cap < s->len + n + 1)
      cap *= 2;
    char *data = realloc(s->data, cap);
    if (!data)
      return 0;
    s->data = data;
    s->cap = cap;
  }
  if (p && n)
    memcpy(s->data + s->len, p, n);
  s->len += n;
  s->data[s->len] = '\0';
  return 1;
}
static int appendf(string *s, const char *format, ...) {
  va_list args;
  va_start(args, format);
  va_list copy;
  va_copy(copy, args);
  int n = vsnprintf(NULL, 0, format, copy);
  va_end(copy);
  if (n < 0 || !append(s, NULL, (size_t)n)) {
    va_end(args);
    return 0;
  }
  vsnprintf(s->data + s->len - (size_t)n, (size_t)n + 1, format, args);
  va_end(args);
  return 1;
}
static int json_string(string *out, const char *text) {
  if (!append(out, "\"", 1))
    return 0;
  for (; *text; ++text) {
    unsigned char c = (unsigned char)*text;
    if (c == '"' || c == '\\') {
      if (!append(out, "\\", 1) || !append(out, text, 1))
        return 0;
    } else if (c == '\n') {
      if (!append(out, "\\n", 2))
        return 0;
    } else if (c == '\r') {
      if (!append(out, "\\r", 2))
        return 0;
    } else if (c == '\t') {
      if (!append(out, "\\t", 2))
        return 0;
    } else if (c < 0x20) {
      if (!appendf(out, "\\u%04x", c))
        return 0;
    } else if (!append(out, text, 1))
      return 0;
  }
  return append(out, "\"", 1);
}
static int response(int fd, int status, const char *type, const char *body) {
  const char *reason = status == 200   ? "OK"
                       : status == 400 ? "Bad Request"
                       : status == 404 ? "Not Found"
                       : status == 413 ? "Payload Too Large"
                                       : "Internal Server Error";
  char header[512];
  int n = snprintf(header, sizeof(header),
                   "HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: "
                   "%zu\r\nConnection: close\r\n\r\n",
                   status, reason, type, strlen(body));
  return n > 0 && send_all(fd, header, (size_t)n) &&
         send_all(fd, body, strlen(body));
}
static void error_response(int fd, int status, const char *message) {
  string body = {0};
  append(&body, LIT("{\"error\":{\"message\":"));
  json_string(&body, message);
  append(&body, LIT(",\"type\":\"invalid_request_error\"}}"));
  response(fd, status, "application/json", body.data);
  free(body.data);
}
static int token_text(const dsm_tokenizer *tokenizer, int32_t id, string *out) {
  FILE *file = tmpfile();
  if (!file)
    return 0;
  int ok = dsm_token_decode(tokenizer, id, file);
  if (ok && fseek(file, 0, SEEK_END) == 0) {
    long n = ftell(file);
    if (n >= 0 && fseek(file, 0, SEEK_SET) == 0) {
      char *buf = n ? malloc((size_t)n) : NULL;
      if ((!n || buf) && (!n || fread(buf, 1, (size_t)n, file) == (size_t)n))
        ok = append(out, buf, (size_t)n);
      else
        ok = 0;
      free(buf);
    } else
      ok = 0;
  } else
    ok = 0;
  fclose(file);
  return ok;
}
static size_t utf8_complete(const char *s, size_t n) {
  if (!n)
    return 0;
  size_t p = n;
  while (p && ((unsigned char)s[p - 1] & 0xc0) == 0x80)
    --p;
  if (p == n)
    return n;
  unsigned char c = (unsigned char)s[p - 1];
  size_t need = c < 0x80 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : 4;
  return n - (p - 1) >= need ? n : p - 1;
}
static size_t stop_at(const char *text, char **stops, size_t count) {
  if (!strcmp(text, "<|im_end|>"))
    return 0;
  size_t best = SIZE_MAX;
  for (size_t i = 0; i < count; ++i) {
    char *at = *stops[i] ? strstr(text, stops[i]) : NULL;
    if (at && (size_t)(at - text) < best)
      best = (size_t)(at - text);
  }
  return best;
}
static int parse_content(const dsm_json *content, string *out) {
  if (!content)
    return 0;
  if (content->type == DSM_JSON_STRING) {
    char *text = dsm_json_string_dup(content);
    int ok = text && append(out, text, strlen(text));
    free(text);
    return ok;
  }
  if (content->type != DSM_JSON_ARRAY)
    return 0;
  for (size_t i = 0; i < content->count; ++i) {
    const dsm_json *part = content->items[i];
    const dsm_json *type = dsm_json_object_get(part, "type");
    const dsm_json *text = dsm_json_object_get(part, "text");
    if (!type || !text || type->type != DSM_JSON_STRING || type->len != 4 ||
        memcmp(type->text, "text", 4) || !parse_content(text, out))
      return 0;
  }
  return 1;
}
static int json_is(const dsm_json *value, const char *text) {
  return value && value->type == DSM_JSON_STRING &&
         value->len == strlen(text) && !memcmp(value->text, text, value->len);
}
static int append_value_text(string *out, const dsm_json *value) {
  char *text = NULL;
  int ok;
  if (!value)
    return 0;
  if (value->type == DSM_JSON_STRING)
    return parse_content(value, out);
  text = dsm_json_dump(value);
  ok = text && append(out, text, strlen(text));
  free(text);
  return ok;
}
static int append_extra_keys(string *out, const dsm_json *object,
                             const char *const *handled, size_t handled_count) {
  if (!object || object->type != DSM_JSON_OBJECT)
    return 1;
  for (size_t i = 0; i + 1 < object->count; i += 2) {
    int skip = 0;
    for (size_t j = 0; j < handled_count; ++j)
      if (json_is(object->items[i], handled[j]))
        skip = 1;
    if (!skip && (!append(out, LIT("\n<")) ||
                  !append(out, object->items[i]->text, object->items[i]->len) ||
                  !append(out, LIT(">")) ||
                  !append_value_text(out, object->items[i + 1]) ||
                  !append(out, LIT("</")) ||
                  !append(out, object->items[i]->text, object->items[i]->len) ||
                  !append(out, LIT(">"))))
      return 0;
  }
  return 1;
}
static int append_tools(string *prompt, const dsm_json *tools) {
  static const char *const parameter_keys[] = {"name", "type", "description"};
  static const char *const tool_keys[] = {"type", "name", "description",
                                          "parameters"};
  static const char *const parameters_keys[] = {"type", "properties"};
  if (!tools || tools->type != DSM_JSON_ARRAY || !tools->count)
    return 1;
  if (!append(prompt, LIT("\n\n# Tools\n\nYou have access to the following "
                          "functions:\n\n<tools>")))
    return 0;
  for (size_t i = 0; i < tools->count; ++i) {
    const dsm_json *tool = tools->items[i],
                   *function = dsm_json_object_get(tool, "function"), *name,
                   *description, *parameters, *properties;
    if (function)
      tool = function;
    name = dsm_json_object_get(tool, "name");
    description = dsm_json_object_get(tool, "description");
    parameters = dsm_json_object_get(tool, "parameters");
    properties = dsm_json_object_get(parameters, "properties");
    if (!name || name->type != DSM_JSON_STRING ||
        !append(prompt, LIT("\n<function>\n<name>")) ||
        !append(prompt, name->text, name->len) ||
        !append(prompt, LIT("</name>")))
      return 0;
    if (description && (!append(prompt, LIT("\n<description>")) ||
                        !append_value_text(prompt, description) ||
                        !append(prompt, LIT("</description>"))))
      return 0;
    if (!append(prompt, LIT("\n<parameters>")))
      return 0;
    if (properties && properties->type == DSM_JSON_OBJECT)
      for (size_t j = 0; j + 1 < properties->count; j += 2) {
        const dsm_json *fields = properties->items[j + 1],
                       *type = dsm_json_object_get(fields, "type"),
                       *desc = dsm_json_object_get(fields, "description");
        if (!append(prompt, LIT("\n<parameter>\n<name>")) ||
            !append(prompt, properties->items[j]->text,
                    properties->items[j]->len) ||
            !append(prompt, LIT("</name>")) ||
            (type && (!append(prompt, LIT("\n<type>")) ||
                      !append_value_text(prompt, type) ||
                      !append(prompt, LIT("</type>")))) ||
            (desc && (!append(prompt, LIT("\n<description>")) ||
                      !append_value_text(prompt, desc) ||
                      !append(prompt, LIT("</description>")))) ||
            !append_extra_keys(prompt, fields, parameter_keys, 3) ||
            !append(prompt, LIT("\n</parameter>")))
          return 0;
      }
    if (!append_extra_keys(prompt, parameters, parameters_keys, 2) ||
        !append(prompt, LIT("\n</parameters>")) ||
        !append_extra_keys(prompt, tool, tool_keys, 4) ||
        !append(prompt, LIT("\n</function>")))
      return 0;
  }
  return append(
      prompt,
      LIT("\n</tools>\n\nIf you choose to call a function ONLY reply in the "
          "following format with NO "
          "suffix:\n\n<tool_call>\n<function=example_function_name>\n<"
          "parameter=example_parameter_1>\nvalue_1\n</"
          "parameter>\n<parameter=example_parameter_2>\nThis is the value for "
          "the second parameter\nthat can span\nmultiple "
          "lines\n</parameter>\n</function>\n</"
          "tool_call>\n\n<IMPORTANT>\nReminder:\n- Function calls MUST follow "
          "the specified format: an inner <function=...></function> block must "
          "be nested within <tool_call></tool_call> XML tags\n- Required "
          "parameters MUST be specified\n- You may provide optional reasoning "
          "for your function call in natural language BEFORE the function "
          "call, but NOT after\n- If there is no function call available, "
          "answer the question like normal with your current knowledge and do "
          "not tell the user about function calls\n</IMPORTANT>"));
}
static int append_tool_calls(string *prompt, const dsm_json *calls) {
  if (!calls || calls->type != DSM_JSON_ARRAY)
    return 0;
  for (size_t i = 0; i < calls->count; ++i) {
    const dsm_json *call = calls->items[i],
                   *function = dsm_json_object_get(call, "function"), *name,
                   *arguments;
    dsm_json *parsed = NULL;
    char error[64], *argument_text = NULL;
    if (function)
      call = function;
    name = dsm_json_object_get(call, "name");
    arguments = dsm_json_object_get(call, "arguments");
    if (!name || name->type != DSM_JSON_STRING ||
        !append(prompt, LIT("\n<tool_call>\n<function=")) ||
        !append(prompt, name->text, name->len) || !append(prompt, LIT(">\n")))
      return 0;
    if (arguments && arguments->type == DSM_JSON_STRING) {
      argument_text = dsm_json_string_dup(arguments);
      if (argument_text)
        dsm_json_parse(argument_text, strlen(argument_text), &parsed, error,
                       sizeof(error));
    }
    if (parsed && parsed->type == DSM_JSON_OBJECT)
      for (size_t j = 0; j + 1 < parsed->count; j += 2)
        if (!append(prompt, LIT("<parameter=")) ||
            !append(prompt, parsed->items[j]->text, parsed->items[j]->len) ||
            !append(prompt, LIT(">\n")) ||
            !append_value_text(prompt, parsed->items[j + 1]) ||
            !append(prompt, LIT("\n</parameter>\n"))) {
          free(argument_text);
          dsm_json_free(parsed);
          return 0;
        }
    free(argument_text);
    dsm_json_free(parsed);
    if (!append(prompt, LIT("</function>\n</tool_call>")))
      return 0;
  }
  return 1;
}
static int thinking_disabled(const dsm_json *root) {
  const dsm_json *kwargs = dsm_json_object_get(root, "chat_template_kwargs");
  const dsm_json *thinking =
      kwargs ? dsm_json_object_get(kwargs, "enable_thinking") : NULL;
  return (thinking && thinking->type == DSM_JSON_BOOL && !thinking->boolean) ||
         json_is(dsm_json_object_get(root, "reasoning_effort"), "none");
}
static int make_prompt(const dsm_json *root, string *prompt) {
  const dsm_json *messages = dsm_json_object_get(root, "messages"),
                 *tools = dsm_json_object_get(root, "tools");
  int tools_present = tools && tools->type == DSM_JSON_ARRAY && tools->count;
  if (!messages || messages->type != DSM_JSON_ARRAY)
    return 0;
  size_t first = 0;
  const dsm_json *system = messages->count ? messages->items[0] : NULL;
  if (system && json_is(dsm_json_object_get(system, "role"), "system")) {
    const dsm_json *content = dsm_json_object_get(system, "content");
    if (!content || !append(prompt, LIT("<|im_start|>system\n")) ||
        !parse_content(content, prompt) || !append_tools(prompt, tools) ||
        !append(prompt, LIT("<|im_end|>\n")))
      return 0;
    first = 1;
  } else if (tools_present &&
             (!append(prompt, LIT("<|im_start|>system\nYou are Qwen, a helpful "
                                  "AI assistant that can interact with a "
                                  "computer to solve tasks.")) ||
              !append_tools(prompt, tools) ||
              !append(prompt, LIT("<|im_end|>\n"))))
    return 0;
  for (size_t i = first; i < messages->count; ++i) {
    const dsm_json *message = messages->items[i],
                   *role = dsm_json_object_get(message, "role"),
                   *content = dsm_json_object_get(message, "content"),
                   *calls = dsm_json_object_get(message, "tool_calls");
    if (!role || role->type != DSM_JSON_STRING)
      return 0;
    if (json_is(role, "assistant") && calls && calls->type == DSM_JSON_ARRAY) {
      if (!append(prompt, LIT("<|im_start|>assistant")) ||
          (content && (!append(prompt, LIT("\n")) ||
                       !append_assistant_content(content, prompt) ||
                       !append(prompt, LIT("\n")))) ||
          !append_tool_calls(prompt, calls) ||
          !append(prompt, LIT("<|im_end|>\n")))
        return 0;
      continue;
    }
    if (json_is(role, "tool")) {
      int open =
          i == first ||
          !json_is(dsm_json_object_get(messages->items[i - 1], "role"), "tool");
      int close =
          i + 1 == messages->count ||
          !json_is(dsm_json_object_get(messages->items[i + 1], "role"), "tool");
      if (!content || (open && !append(prompt, LIT("<|im_start|>user\n"))) ||
          !append(prompt, LIT("<tool_response>\n")) ||
          !parse_content(content, prompt) ||
          !append(prompt, LIT("\n</tool_response>\n")) ||
          (close && !append(prompt, LIT("<|im_end|>\n"))))
        return 0;
      continue;
    }
    if (!content ||
        (!json_is(role, "system") && !json_is(role, "user") &&
         !json_is(role, "assistant")) ||
        !append(prompt, LIT("<|im_start|>")) ||
        !append(prompt, role->text, role->len) || !append(prompt, LIT("\n")) ||
        !(json_is(role, "assistant") ? append_assistant_content(content, prompt)
                                     : parse_content(content, prompt)) ||
        !append(prompt, LIT("<|im_end|>\n")))
      return 0;
  }
  if (!append(prompt, LIT("<|im_start|>assistant\n")))
    return 0;
  if (thinking_disabled(root))
    return append(prompt, LIT("<think>\n\n</think>\n\n"));
  return append(prompt, LIT("<think>\n"));
}
int dsm_server_make_prompt(const dsm_json *root, char **out) {
  string prompt = {0};
  *out = NULL;
  if (!make_prompt(root, &prompt)) {
    free(prompt.data);
    return 0;
  }
  *out = prompt.data;
  return 1;
}
static char *trim_copy(const char *text, size_t len) {
  while (len &&
         (*text == ' ' || *text == '\t' || *text == '\r' || *text == '\n')) {
    ++text;
    --len;
  }
  while (len && (text[len - 1] == ' ' || text[len - 1] == '\t' ||
                 text[len - 1] == '\r' || text[len - 1] == '\n'))
    --len;
  char *out = malloc(len + 1);
  if (out) {
    memcpy(out, text, len);
    out[len] = '\0';
  }
  return out;
}
void dsm_tool_calls_free(dsm_tool_call *calls, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    free(calls[i].name);
    free(calls[i].arguments);
  }
  free(calls);
}
static int parse_one_tool_call(const char *text, size_t len,
                               dsm_tool_call *call) {
  const char *function = strstr(text, "<function="), *function_end, *name_end;
  string arguments = {0};
  if (!function || function >= text + len ||
      !(name_end = strchr(function + 10, '>')) || name_end >= text + len ||
      !(function_end = strstr(name_end + 1, "</function>")) ||
      function_end > text + len)
    return 0;
  call->name = trim_copy(function + 10, (size_t)(name_end - (function + 10)));
  if (!call->name || !*call->name || !append(&arguments, LIT("{")))
    goto bad;
  const char *at = name_end + 1;
  int first = 1;
  while ((at = strstr(at, "<parameter=")) && at < function_end) {
    const char *parameter_end = strchr(at + 11, '>'), *value_end;
    char *parameter, *value;
    dsm_json *parsed = NULL;
    char error[64];
    if (!parameter_end || parameter_end >= function_end ||
        !(value_end = strstr(parameter_end + 1, "</parameter>")) ||
        value_end > function_end)
      goto bad;
    parameter = trim_copy(at + 11, (size_t)(parameter_end - (at + 11)));
    value =
        trim_copy(parameter_end + 1, (size_t)(value_end - (parameter_end + 1)));
    if (!parameter || !value || !*parameter ||
        (!first && !append(&arguments, LIT(","))) ||
        !json_string(&arguments, parameter) || !append(&arguments, LIT(":"))) {
      free(parameter);
      free(value);
      goto bad;
    }
    if (dsm_json_parse(value, strlen(value), &parsed, error, sizeof(error))) {
      char *dump = dsm_json_dump(parsed);
      int ok = dump && append(&arguments, dump, strlen(dump));
      free(dump);
      dsm_json_free(parsed);
      if (!ok) {
        free(parameter);
        free(value);
        goto bad;
      }
    } else if (!json_string(&arguments, value)) {
      free(parameter);
      free(value);
      goto bad;
    }
    free(parameter);
    free(value);
    first = 0;
    at = value_end + strlen("</parameter>");
  }
  if (!append(&arguments, LIT("}")))
    goto bad;
  call->arguments = arguments.data;
  return 1;
bad:
  free(call->name);
  call->name = NULL;
  free(arguments.data);
  return 0;
}
int dsm_server_parse_tool_calls(const char *output, char **content,
                                dsm_tool_call **calls, size_t *count) {
  const char *at = strstr(output, "<tool_call>");
  string text = {0};
  *content = NULL;
  *calls = NULL;
  *count = 0;
  if (!at) {
    *content = strdup(output);
    return *content != NULL;
  }
  if (!append(&text, output, (size_t)(at - output)))
    goto bad;
  while (at) {
    const char *end = strstr(at, "</tool_call>");
    dsm_tool_call call = {0}, *new_calls;
    if (!end || !parse_one_tool_call(at, (size_t)(end + 12 - at), &call))
      goto bad;
    new_calls = realloc(*calls, (*count + 1) * sizeof(**calls));
    if (!new_calls) {
      free(call.name);
      free(call.arguments);
      goto bad;
    }
    *calls = new_calls;
    (*calls)[(*count)++] = call;
    at = strstr(end + 12, "<tool_call>");
  }
  *content = trim_copy(text.data ? text.data : "", text.len);
  free(text.data);
  if (*content)
    return 1;
bad:
  free(text.data);
  free(*content);
  *content = NULL;
  dsm_tool_calls_free(*calls, *count);
  *calls = NULL;
  *count = 0;
  return 0;
}
static int split_reasoning(const char *output, char **reasoning,
                           char **content) {
  const char *start = strstr(output, "<think>");
  const char *end = strstr(output, "</think>");
  string visible = {0};
  *reasoning = NULL;
  if (!start) {
    if (end) {
      *reasoning = trim_copy(output, (size_t)(end - output));
      *content = trim_copy(end + strlen("</think>"),
                           strlen(end + strlen("</think>")));
      return *reasoning && *content;
    }
    *content = strdup(output);
    return *content != NULL;
  }
  end = strstr(start + strlen("<think>"), "</think>");
  if (!end) {
    *reasoning =
        trim_copy(start + strlen("<think>"), strlen(start + strlen("<think>")));
    *content = trim_copy(output, (size_t)(start - output));
    return *reasoning && *content;
  }
  *reasoning = trim_copy(start + strlen("<think>"),
                         (size_t)(end - (start + strlen("<think>"))));
  if (!append(&visible, output, (size_t)(start - output)) ||
      !append(&visible, end + strlen("</think>"),
              strlen(end + strlen("</think>")))) {
    free(visible.data);
    free(*reasoning);
    *reasoning = NULL;
    return 0;
  }
  *content = trim_copy(visible.data, visible.len);
  free(visible.data);
  return *reasoning && *content;
}
static int append_assistant_content(const dsm_json *value, string *out) {
  string raw = {0};
  char *reasoning = NULL;
  char *content = NULL;
  int ok = parse_content(value, &raw) &&
           split_reasoning(raw.data, &reasoning, &content) &&
           append(out, content, strlen(content));
  free(raw.data);
  free(reasoning);
  free(content);
  return ok;
}
static size_t common_prefix(const int32_t *a, size_t an, const int32_t *b,
                            size_t bn) {
  size_t n = an < bn ? an : bn, i = 0;
  while (i < n && a[i] == b[i])
    ++i;
  return i;
}
static size_t checkpoint_split(const int32_t *tokens, size_t len, size_t start,
                               int32_t im_start) {
  /* Reasoning stripping commonly diverges in the assistant header. Checkpoint
     immediately before its final im_start token, falling back to len - 1 so
     a fully reused prompt can re-evaluate its last token from a saved state. */
  if (im_start >= 0)
    for (size_t i = len; i-- > start;)
      if (tokens[i] == im_start)
        return i;
  return len > start ? len - 1 : start;
}
static int cache_tokens(server *s, const dsm_tokens *tokens,
                        const int32_t *generated, size_t generated_len) {
  size_t n = tokens->len + generated_len;
  if (n > s->cached_cap) {
    int32_t *cache = realloc(s->cached, n * sizeof(*cache));
    if (!cache)
      return 0;
    s->cached = cache;
    s->cached_cap = n;
  }
  memcpy(s->cached, tokens->ids, tokens->len * sizeof(*s->cached));
  if (generated_len)
    memcpy(s->cached + tokens->len, generated,
           generated_len * sizeof(*s->cached));
  s->cached_len = n;
  return 1;
}
static int sse_chunk(int fd, const char *id, const char *content,
                     const char *reasoning, const char *finish) {
  string out = {0};
  if (!append(&out, LIT("data: {\"id\":\"")) || !append(&out, id, strlen(id)) ||
      !append(&out, LIT("\",\"object\":\"chat.completion.chunk\",\"choices\":[{"
                        "\"index\":0,\"delta\":{")))
    goto bad;
  if (content) {
    if (!append(&out, LIT("\"content\":")) || !json_string(&out, content))
      goto bad;
  }
  if (reasoning) {
    if (content && !append(&out, LIT(",")))
      goto bad;
    if (!append(&out, LIT("\"reasoning_content\":")) ||
        !json_string(&out, reasoning))
      goto bad;
  }
  if (!append(&out, LIT("},\"finish_reason\":")))
    goto bad;
  if (finish) {
    if (!json_string(&out, finish))
      goto bad;
  } else if (!append(&out, LIT("null")))
    goto bad;
  if (!append(&out, LIT("}]}\n\n")))
    goto bad;
  int ok = send_all(fd, out.data, out.len);
  free(out.data);
  return ok;
bad:
  free(out.data);
  return 0;
}
static int sse_tool_chunks(int fd, const char *id, const dsm_tool_call *calls,
                           size_t count) {
  for (size_t i = 0; i < count; ++i) {
    string out = {0};
    if (!append(&out, LIT("data: {\"id\":\"")) ||
        !append(&out, id, strlen(id)) ||
        !append(&out,
                LIT("\",\"object\":\"chat.completion.chunk\",\"choices\":[{"
                    "\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":")) ||
        !appendf(&out, "%zu", i) || !append(&out, LIT(",\"id\":\"call_")) ||
        !appendf(&out, "%zu", i) ||
        !append(&out,
                LIT("\",\"type\":\"function\",\"function\":{\"name\":")) ||
        !json_string(&out, calls[i].name) ||
        !append(&out, LIT(",\"arguments\":")) ||
        !json_string(&out, calls[i].arguments) ||
        !append(&out, LIT("}}]},\"finish_reason\":null}]}\n\n"))) {
      free(out.data);
      return 0;
    }
    int ok = send_all(fd, out.data, out.len);
    free(out.data);
    if (!ok)
      return 0;
  }
  return 1;
}
static int generate(server *s, int fd, const dsm_json *root) {
  string prompt = {0}, output = {0}, pending = {0};
  dsm_tokens tokens = {0};
  int32_t *generated = NULL;
  size_t generated_len = 0, generated_cap = 0, reused = 0;
  char **stops = NULL;
  size_t stop_count = 0;
  char *content = NULL;
  char *reasoning = NULL;
  dsm_tool_call *calls = NULL;
  size_t call_count = 0;
  int tools_requested = 0;
  int streamed_reasoning = 0;
  int thinking = !thinking_disabled(root);
  dsm_sampler sampler = {0};
  const dsm_json *v = dsm_json_object_get(root, "max_completion_tokens");
  if (!v)
    v = dsm_json_object_get(root, "max_tokens");
  int max_tokens = v && v->type == DSM_JSON_NUMBER ? (int)v->number : 16;
  const dsm_json *temp_v = dsm_json_object_get(root, "temperature"),
                 *top_p_v = dsm_json_object_get(root, "top_p"),
                 *top_k_v = dsm_json_object_get(root, "top_k"),
                 *min_p_v = dsm_json_object_get(root, "min_p"),
                 *seed_v = dsm_json_object_get(root, "seed"),
                 *presence_v = dsm_json_object_get(root, "presence_penalty"),
                 *frequency_v = dsm_json_object_get(root, "frequency_penalty"),
                 *stream_v = dsm_json_object_get(root, "stream"),
                 *stop_v = dsm_json_object_get(root, "stop");
  float temp =
      temp_v && temp_v->type == DSM_JSON_NUMBER ? (float)temp_v->number : 1.0f;
  float top_p = top_p_v && top_p_v->type == DSM_JSON_NUMBER
                    ? (float)top_p_v->number
                    : 1.0f;
  uint32_t top_k =
      top_k_v && top_k_v->type == DSM_JSON_NUMBER && top_k_v->number > 0
          ? (uint32_t)top_k_v->number
          : 0;
  float min_p =
      min_p_v && min_p_v->type == DSM_JSON_NUMBER ? (float)min_p_v->number : 0;
  uint64_t seed =
      seed_v && seed_v->type == DSM_JSON_NUMBER && seed_v->number > 0
          ? (uint64_t)seed_v->number
          : 1;
  float presence_penalty = presence_v && presence_v->type == DSM_JSON_NUMBER
                               ? (float)presence_v->number
                               : 0;
  float frequency_penalty = frequency_v && frequency_v->type == DSM_JSON_NUMBER
                                ? (float)frequency_v->number
                                : 0;
  int stream = stream_v && stream_v->type == DSM_JSON_BOOL && stream_v->boolean;
  if (max_tokens < 0 || !make_prompt(root, &prompt) ||
      !dsm_tokenize(s->tokenizer, prompt.data, 0, &tokens)) {
    error_response(fd, 400, "invalid messages or max_tokens");
    goto done;
  }
  if (tokens.len > s->context) {
    char message[128];
    snprintf(message, sizeof(message),
             "prompt is %zu tokens but context length is %u", tokens.len,
             s->context);
    error_response(fd, 400, message);
    goto done;
  }
  if ((size_t)max_tokens > s->context - tokens.len)
    max_tokens = (int)(s->context - tokens.len);
  tools_requested =
      dsm_json_object_get(root, "tools") &&
      dsm_json_object_get(root, "tools")->type == DSM_JSON_ARRAY &&
      dsm_json_object_get(root, "tools")->count;
  if (stop_v) {
    if (stop_v->type == DSM_JSON_STRING)
      stop_count = 1;
    else if (stop_v->type == DSM_JSON_ARRAY)
      stop_count = stop_v->count;
    else {
      error_response(fd, 400, "stop must be a string or array");
      goto done;
    }
    stops = calloc(stop_count, sizeof(*stops));
    if (!stops)
      goto done;
    for (size_t i = 0; i < stop_count; ++i) {
      stops[i] = dsm_json_string_dup(
          stop_v->type == DSM_JSON_STRING ? stop_v : stop_v->items[i]);
      if (!stops[i]) {
        error_response(fd, 400, "stop must contain strings");
        goto done;
      }
    }
  }
  reused = common_prefix(s->cached, s->cached_len, tokens.ids, tokens.len);
  /* The cached logits follow the last generated token, not the last prompt
     token. Re-evaluate that prompt token when the entire prompt is reused. */
  if (reused == tokens.len && reused)
    --reused;
  size_t requested_reuse = reused;
  int recurrent = s->model.is_recurrent != NULL;
  if (s->use_metal) {
    uint32_t restored = dsm_metal_rewind(s->metal, (uint32_t)reused);
    if (restored == UINT32_MAX) {
      error_response(fd, 500, "cache state error");
      goto done;
    }
    reused = restored;
  }
  if (!s->use_metal && !dsm_forward_truncate(s->forward, (uint32_t)reused)) {
    error_response(fd, 500, "cache state error");
    goto done;
  }
  fprintf(
      stderr,
      "dsm: received prompt tokens %zu, reuse requested %zu, restored %zu\n",
      tokens.len, requested_reuse, reused);
  double prefill_started = now();
  if (s->use_metal && recurrent) {
    size_t split = checkpoint_split(tokens.ids, tokens.len, reused,
                                    dsm_token_im_start(s->tokenizer));
    if (reused < split &&
        !dsm_metal_prefill(s->metal, tokens.ids + reused,
                           (uint32_t)(split - reused), s->logits)) {
      error_response(fd, 500, "prompt inference failed");
      goto done;
    }
    if (reused < split && !dsm_metal_checkpoint(s->metal)) {
      error_response(fd, 500, "checkpoint failed");
      goto done;
    }
    if (split < tokens.len &&
        !dsm_metal_prefill(s->metal, tokens.ids + split,
                           (uint32_t)(tokens.len - split), s->logits)) {
      error_response(fd, 500, "prompt inference failed");
      goto done;
    }
    if (!dsm_metal_checkpoint(s->metal)) {
      error_response(fd, 500, "checkpoint failed");
      goto done;
    }
  } else if (s->use_metal) {
    if (reused < tokens.len &&
        !dsm_metal_prefill(s->metal, tokens.ids + reused,
                           (uint32_t)(tokens.len - reused), s->logits)) {
      error_response(fd, 500, "prompt inference failed");
      goto done;
    }
  } else
    for (size_t i = reused; i < tokens.len; ++i)
      if (!dsm_forward_token(s->forward, tokens.ids[i], s->logits)) {
        error_response(fd, 500, "prompt inference failed");
        goto done;
      }
  double prefill_elapsed = now() - prefill_started;
  char request_id[64];
  snprintf(request_id, sizeof(request_id), "chatcmpl-%lx",
           (unsigned long)time(NULL));
  if (stream && !send_all(fd,
                          "HTTP/1.1 200 OK\r\nContent-Type: "
                          "text/event-stream\r\nCache-Control: "
                          "no-cache\r\nConnection: close\r\n\r\n",
                          91))
    goto done;
  dsm_sampler_init(&sampler, temp, top_p, top_k, min_p, seed, presence_penalty,
                   frequency_penalty);
  double decode_started = now();
  int finish_length = 1, streaming_tool = 0;
  for (int i = 0; i < max_tokens; ++i) {
    int32_t id = dsm_sample(&sampler, s->logits, s->model.vocab);
    string piece = {0};
    if (dsm_token_is_im_end(s->tokenizer, id)) {
      finish_length = 0;
      break;
    }
    if (!token_text(s->tokenizer, id, &piece) ||
        !(s->use_metal ? dsm_metal_token(s->metal, id, s->logits)
                       : dsm_forward_token(s->forward, id, s->logits))) {
      free(piece.data);
      break;
    }
    if (generated_len == generated_cap) {
      size_t cap = generated_cap ? generated_cap * 2 : 32;
      int32_t *p = realloc(generated, cap * sizeof(*p));
      if (!p) {
        free(piece.data);
        break;
      }
      generated = p;
      generated_cap = cap;
    }
    generated[generated_len++] = id;
    if (!append(&output, piece.data, piece.len) ||
        !append(&pending, piece.data, piece.len) ||
        output.len > DSM_MAX_OUTPUT) {
      free(piece.data);
      break;
    }
    free(piece.data);
    size_t stop = stop_at(output.data, stops, stop_count);
    if (stop != SIZE_MAX) {
      output.len = stop;
      output.data[stop] = '\0';
      finish_length = 0;
      break;
    }
    size_t ready = utf8_complete(pending.data, pending.len);
    if (stream && ready) {
      size_t send = ready;
      if (tools_requested && !streaming_tool) {
        char *tool = strstr(output.data, "<tool_call>");
        if (tool) {
          send = (size_t)(tool - output.data);
          streaming_tool = 1;
        } else
          send = ready > sizeof("<tool_call>") - 1
                     ? utf8_complete(pending.data,
                                     ready - (sizeof("<tool_call>") - 1))
                     : 0;
      }
      if (send) {
        char saved = pending.data[send];
        pending.data[send] = '\0';
        int ok = sse_chunk(fd, request_id, NULL, pending.data, NULL);
        streamed_reasoning = 1;
        pending.data[send] = saved;
        if (!ok)
          goto logged;
        memmove(pending.data, pending.data + send, pending.len - send);
        pending.len -= send;
        pending.data[pending.len] = '\0';
      }
    }
  }
  if (!dsm_server_parse_tool_calls(output.data ? output.data : "", &content,
                                   &calls, &call_count)) {
    error_response(fd, 500, "failed to parse generated tool call");
    goto logged;
  }
  if (thinking) {
    if (!strstr(content, "<think>") && !strstr(content, "</think>")) {
      reasoning = content;
      content = strdup("");
    } else if (!split_reasoning(content, &reasoning, &content)) {
      error_response(fd, 500, "failed to separate reasoning");
      goto logged;
    }
  }
  if (stream) {
    if ((thinking && !streamed_reasoning && reasoning && *reasoning &&
         !sse_chunk(fd, request_id, NULL, reasoning, NULL)) ||
        (thinking && content && *content &&
         !sse_chunk(fd, request_id, content, NULL, NULL)) ||
        (tools_requested && !thinking && !streaming_tool && pending.len &&
         !sse_chunk(fd, request_id, pending.data, NULL, NULL)) ||
        (call_count && !sse_tool_chunks(fd, request_id, calls, call_count)) ||
        (!thinking && !tools_requested && pending.len &&
         !sse_chunk(fd, request_id, pending.data, NULL, NULL)) ||
        !sse_chunk(fd, request_id, NULL, NULL,
                   call_count      ? "tool_calls"
                   : finish_length ? "length"
                                   : "stop") ||
        !send_all(fd, LIT("data: [DONE]\n\n")))
      goto logged;
  } else {
    string body = {0};
    append(&body, LIT("{\"id\":\""));
    append(&body, request_id, strlen(request_id));
    append(&body,
           LIT("\",\"object\":\"chat.completion\",\"choices\":[{\"index\":0,"
               "\"message\":{\"role\":\"assistant\",\"content\":"));
    json_string(&body, content);
    if (reasoning && *reasoning) {
      append(&body, LIT(",\"reasoning_content\":"));
      json_string(&body, reasoning);
    }
    if (call_count) {
      append(&body, LIT(",\"tool_calls\":["));
      for (size_t i = 0; i < call_count; ++i) {
        if (i)
          append(&body, LIT(","));
        appendf(&body,
                "{\"id\":\"call_%zu\",\"type\":\"function\",\"function\":{"
                "\"name\":",
                i);
        json_string(&body, calls[i].name);
        append(&body, LIT(",\"arguments\":"));
        json_string(&body, calls[i].arguments);
        append(&body, LIT("}}"));
      }
      append(&body, LIT("]"));
    }
    appendf(&body,
            "},\"finish_reason\":\"%s\"}],\"usage\":{\"prompt_tokens\":%zu,"
            "\"completion_tokens\":%zu,\"total_tokens\":%zu}}",
            call_count      ? "tool_calls"
            : finish_length ? "length"
                            : "stop",
            tokens.len, generated_len, tokens.len + generated_len);
    response(fd, 200, "application/json", body.data);
    free(body.data);
  }
logged: {
  double elapsed = now() - decode_started;
  fprintf(stderr,
          "dsm: prompt tokens %zu, reused tokens %zu, prefill %.2f tok/s, "
          "decode %.2f tok/s\n",
          tokens.len, reused,
          (tokens.len - reused) /
              (prefill_elapsed > 0 ? prefill_elapsed : 1e-9),
          generated_len / (elapsed > 0 ? elapsed : 1e-9));
  cache_tokens(s, &tokens, generated, generated_len);
}
done:
  dsm_sampler_free(&sampler);
  dsm_tool_calls_free(calls, call_count);
  free(content);
  free(reasoning);
  for (size_t i = 0; i < stop_count; ++i)
    free(stops[i]);
  free(stops);
  free(generated);
  dsm_tokens_free(&tokens);
  free(prompt.data);
  free(output.data);
  free(pending.data);
  return 1;
}
static int read_request(int fd, char **method, char **path, char **body) {
  string request = {0};
  char buffer[8192];
  size_t header_len = 0, content_len = 0;
  for (;;) {
    ssize_t n = recv(fd, buffer, sizeof(buffer), 0);
    if (n <= 0 || !append(&request, buffer, (size_t)n) ||
        request.len > DSM_MAX_BODY + 65536)
      goto bad;
    char *end = strstr(request.data, "\r\n\r\n");
    if (!end)
      continue;
    header_len = (size_t)(end + 4 - request.data);
    char request_method[16], request_path[2048];
    if (sscanf(request.data, "%15s %2047s", request_method, request_path) != 2)
      goto bad;
    *method = strdup(request_method);
    *path = strdup(request_path);
    if (!*method || !*path)
      goto bad;
    const char *cl = strcasestr(request.data, "\r\nContent-Length:");
    if (!cl && !strcmp(*method, "POST"))
      goto bad;
    if (cl) {
      char *number_end;
      unsigned long parsed = strtoul(cl + 17, &number_end, 10);
      if (number_end == cl + 17)
        goto bad;
      content_len = (size_t)parsed;
    }
    if (content_len > DSM_MAX_BODY) {
      free(request.data);
      return -1;
    }
    while (request.len < header_len + content_len) {
      n = recv(fd, buffer, sizeof(buffer), 0);
      if (n <= 0 || !append(&request, buffer, (size_t)n))
        goto bad;
    }
    *body = malloc(content_len + 1);
    if (!*body)
      goto bad;
    memcpy(*body, request.data + header_len, content_len);
    (*body)[content_len] = '\0';
    free(request.data);
    return 1;
  }
bad:
  free(request.data);
  free(*method);
  free(*path);
  *method = *path = NULL;
  return 0;
}
static void handle(server *s, int fd) {
  char *method = NULL, *path = NULL, *body = NULL;
  int read = read_request(fd, &method, &path, &body);
  if (read < 0)
    error_response(fd, 413, "request body exceeds 8 MB");
  else if (!read)
    error_response(fd, 400, "invalid HTTP request");
  else if (!strcmp(method, "GET") && !strcmp(path, "/v1/models")) {
    string out = {0};
    append(&out, LIT("{\"object\":\"list\",\"data\":[{\"id\":"));
    json_string(&out, s->model_id);
    append(&out,
           LIT(",\"object\":\"model\",\"owned_by\":\"dwarfstar-mini\"}]}"));
    response(fd, 200, "application/json", out.data);
    free(out.data);
  } else if (!strcmp(method, "POST") && !strcmp(path, "/v1/chat/completions")) {
    dsm_json *root = NULL;
    char error[128];
    if (!dsm_json_parse(body, strlen(body), &root, error, sizeof(error)) ||
        !root || root->type != DSM_JSON_OBJECT)
      error_response(fd, 400,
                     error[0] ? error : "request must be a JSON object");
    else
      generate(s, fd, root);
    dsm_json_free(root);
  } else
    error_response(fd, 404, "not found");
  free(method);
  free(path);
  free(body);
}
static int listener(const char *host, uint16_t port) {
  char port_text[16];
  snprintf(port_text, sizeof(port_text), "%u", port);
  struct addrinfo hints = {0}, *result = NULL;
  hints.ai_family = AF_UNSPEC;
  hints.ai_socktype = SOCK_STREAM;
  hints.ai_flags = AI_PASSIVE;
  if (getaddrinfo(host, port_text, &hints, &result))
    return -1;
  int fd = -1;
  for (struct addrinfo *it = result; it; it = it->ai_next) {
    fd = socket(it->ai_family, it->ai_socktype, it->ai_protocol);
    int yes = 1;
    if (fd >= 0 &&
        setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) == 0 &&
        bind(fd, it->ai_addr, it->ai_addrlen) == 0 && listen(fd, 8) == 0)
      break;
    if (fd >= 0)
      close(fd);
    fd = -1;
  }
  freeaddrinfo(result);
  return fd;
}
int dsm_server_run(const char *model_path, const dsm_server_config *config) {
  server s = {0};
  s.context = config->context;
  s.threads = config->threads;
  if (config->backend && strcmp(config->backend, "cpu") &&
      strcmp(config->backend, "metal")) {
    fprintf(stderr, "dsm: unknown backend '%s'\n", config->backend);
    return 2;
  }
  s.use_metal = config->backend ? !strcmp(config->backend, "metal")
                                : dsm_metal_available();
  if (!dsm_model_open(&s.model, model_path)) {
    fprintf(stderr, "dsm: %s\n", s.model.error);
    return 1;
  }
  if (!s.context)
    s.context = s.model.context < DSM_DEFAULT_CONTEXT ? s.model.context
                                                      : DSM_DEFAULT_CONTEXT;
  if (!dsm_tokenizer_init(&s.tokenizer, &s.model.gguf, s.model.error,
                          sizeof(s.model.error)) ||
      !(s.logits = malloc((size_t)s.model.vocab * sizeof(*s.logits)))) {
    fprintf(stderr, "dsm: server initialization failed\n");
    dsm_tokenizer_free(s.tokenizer);
    dsm_model_close(&s.model);
    return 1;
  }
  if (s.use_metal) {
    char error[256];
    if (!dsm_metal_init(&s.metal, &s.model, s.context, error, sizeof(error))) {
      fprintf(stderr, "dsm: Metal initialization failed: %s\n", error);
      free(s.logits);
      dsm_tokenizer_free(s.tokenizer);
      dsm_model_close(&s.model);
      return 1;
    }
  } else if (!dsm_forward_init(&s.forward, &s.model, s.context, s.threads)) {
    fprintf(stderr, "dsm: cannot allocate CPU forward state\n");
    free(s.logits);
    dsm_tokenizer_free(s.tokenizer);
    dsm_model_close(&s.model);
    return 1;
  }
  const char *base = strrchr(model_path, '/');
  base = base ? base + 1 : model_path;
  snprintf(s.model_id, sizeof(s.model_id), "%s",
           config->alias ? config->alias : base);
  char *suffix = strstr(s.model_id, ".gguf");
  if (!config->alias && suffix && !suffix[5])
    *suffix = '\0';
  int fd = listener(config->host, config->port);
  if (fd < 0) {
    fprintf(stderr, "dsm: cannot listen on %s:%u\n", config->host,
            config->port);
    free(s.logits);
    dsm_metal_free(s.metal);
    dsm_forward_free(s.forward);
    dsm_tokenizer_free(s.tokenizer);
    dsm_model_close(&s.model);
    return 1;
  }
  signal(SIGPIPE, SIG_IGN);
  fprintf(stderr, "dsm: serving %s on http://%s:%u (%s, ctx %u, KV %.2f MiB)\n",
          s.model_id, config->host, config->port, s.use_metal ? "metal" : "cpu",
          s.context,
          (double)s.model.layers * s.context * s.model.kv_heads *
              s.model.key_dim * 2 * sizeof(float) / (1024.0 * 1024.0));
  for (;;) {
    int client = accept(fd, NULL, NULL);
    if (client >= 0) {
      handle(&s, client);
      close(client);
    } else if (errno != EINTR)
      break;
  }
  close(fd);
  free(s.cached);
  free(s.logits);
  dsm_metal_free(s.metal);
  dsm_forward_free(s.forward);
  dsm_tokenizer_free(s.tokenizer);
  dsm_model_close(&s.model);
  return 0;
}
