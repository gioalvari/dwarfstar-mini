#ifndef DSM_SERVER_H
#define DSM_SERVER_H

#include <stdint.h>

typedef struct {
    const char *backend;
    const char *host;
    const char *alias;
    uint16_t port;
    uint32_t context;
    uint32_t threads;
} dsm_server_config;

int dsm_server_run(const char *model_path, const dsm_server_config *config);

#endif
