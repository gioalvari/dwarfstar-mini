#ifndef DSM_JSON_H
#define DSM_JSON_H

#include <stddef.h>

typedef enum { DSM_JSON_NULL, DSM_JSON_BOOL, DSM_JSON_NUMBER, DSM_JSON_STRING,
               DSM_JSON_ARRAY, DSM_JSON_OBJECT } dsm_json_type;
typedef struct dsm_json dsm_json;
struct dsm_json {
    dsm_json_type type;
    const char *text;
    size_t len;
    double number;
    int boolean;
    dsm_json **items;
    size_t count;
};

/* Parses one complete JSON value. The returned tree owns all child arrays. */
int dsm_json_parse(const char *text, size_t len, dsm_json **out, char *error,
                   size_t error_size);
void dsm_json_free(dsm_json *value);
const dsm_json *dsm_json_object_get(const dsm_json *object, const char *key);
char *dsm_json_string_dup(const dsm_json *value);
/* Allocates a compact JSON representation of value. */
char *dsm_json_dump(const dsm_json *value);

#endif
