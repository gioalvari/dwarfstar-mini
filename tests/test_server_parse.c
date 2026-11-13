#include "json.h"
#include "sampler.h"
#include "server.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int contains(const char *text, const char *part) {
    return strstr(text, part) != NULL;
}

int main(void) {
    const float logits[] = {1.0f, 2.0f, 3.0f};
    dsm_sampler sampler;
    dsm_sampler_init(&sampler, 0.8f, 1.0f, 1, 0, 123, 0, 0);
    if (dsm_sample(&sampler, logits, 3) != 2) {
        fprintf(stderr, "top_k sampling failed\n");
        dsm_sampler_free(&sampler);
        return 1;
    }
    dsm_sampler_free(&sampler);
    const char *request =
        "{\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"description\":\"Read a file\",\"parameters\":{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\",\"description\":\"Path\"}}}}}],"
        "\"messages\":[{\"role\":\"user\",\"content\":\"Read README.md\"},{\"role\":\"assistant\",\"content\":\"\",\"tool_calls\":[{\"type\":\"function\",\"function\":{\"name\":\"read_file\",\"arguments\":\"{\\\"path\\\":\\\"README.md\\\"}\"}}]},{\"role\":\"tool\",\"content\":\"# dwarfstar-mini\"}]}";
    dsm_json *root = NULL;
    char error[128], *prompt = NULL, *content = NULL;
    dsm_tool_call *calls = NULL;
    size_t count = 0;
    if (!dsm_json_parse(request, strlen(request), &root, error, sizeof(error)) ||
        !dsm_server_make_prompt(root, &prompt) ||
        !contains(prompt, "<tools>\n<function>\n<name>read_file</name>") ||
        !contains(prompt, "<parameter>\n<name>path</name>\n<type>string</type>") ||
        !contains(prompt, "<function=read_file>\n<parameter=path>\nREADME.md") ||
        !contains(prompt, "<|im_start|>user\n<tool_response>\n# dwarfstar-mini\n</tool_response>\n<|im_end|>")) {
        fprintf(stderr, "prompt render failed: %s\n", prompt ? prompt : error);
        free(prompt); dsm_json_free(root); return 1;
    }
    dsm_json_free(root);
    root = NULL;
    free(prompt);
    prompt = NULL;
    const char *thinking_disabled =
        "{\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"No thought\"}]}";
    if (!dsm_json_parse(thinking_disabled, strlen(thinking_disabled), &root, error,
                        sizeof(error)) ||
        !dsm_server_make_prompt(root, &prompt) ||
        !contains(prompt, "<think>\n\n</think>\n\n")) {
        fprintf(stderr, "thinking-disabled prompt render failed\n");
        free(prompt); dsm_json_free(root); return 1;
    }
    dsm_json_free(root);
    root = NULL;
    free(prompt);
    prompt = NULL;
    const char *reasoning_none =
        "{\"reasoning_effort\":\"none\",\"messages\":[{\"role\":\"user\",\"content\":\"No thought\"}]}";
    if (!dsm_json_parse(reasoning_none, strlen(reasoning_none), &root, error,
                        sizeof(error)) ||
        !dsm_server_make_prompt(root, &prompt) ||
        !contains(prompt, "<think>\n\n</think>\n\n")) {
        fprintf(stderr, "reasoning_effort prompt render failed\n");
        free(prompt); dsm_json_free(root); return 1;
    }
    if (!dsm_server_parse_tool_calls("I will read it.\n<tool_call>\n<function=read_file>\n<parameter=path>\nREADME.md\n</parameter>\n<parameter=offset>\n12\n</parameter>\n<parameter=options>\n{\"raw\":true}\n</parameter>\n</function>\n</tool_call>", &content, &calls, &count) ||
        strcmp(content, "I will read it.") || count != 1 ||
        strcmp(calls[0].name, "read_file") ||
        strcmp(calls[0].arguments, "{\"path\":\"README.md\",\"offset\":12,\"options\":{\"raw\":true}}")) {
        fprintf(stderr, "tool-call parse failed\n");
        free(prompt); free(content); dsm_tool_calls_free(calls, count); dsm_json_free(root); return 1;
    }
    puts("server prompt and tool-call parsing passed");
    free(prompt); free(content); dsm_tool_calls_free(calls, count); dsm_json_free(root);
    return 0;
}
