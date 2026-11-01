#ifndef DSM_SERVER_H
#define DSM_SERVER_H

#include <stdint.h>
#include <stddef.h>

struct dsm_json;

typedef struct {
    char *name;
    char *arguments;
} dsm_tool_call;

typedef struct {
    const char *backend;
    const char *host;
    const char *alias;
    uint16_t port;
    uint32_t context;
    uint32_t threads;
} dsm_server_config;

int dsm_server_run(const char *model_path, const dsm_server_config *config);

/* Helpers shared with the focused server rendering/parser tests. */
int dsm_server_make_prompt(const struct dsm_json *root, char **out);
int dsm_server_parse_tool_calls(const char *output, char **content,
                                dsm_tool_call **calls, size_t *count);
void dsm_tool_calls_free(dsm_tool_call *calls, size_t count);

#endif
