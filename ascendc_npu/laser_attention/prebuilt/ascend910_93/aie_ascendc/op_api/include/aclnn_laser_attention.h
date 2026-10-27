
/*
 * calution: this file was generated automaticlly donot change it.
*/

#ifndef ACLNN_LASER_ATTENTION_H_
#define ACLNN_LASER_ATTENTION_H_

#include "aclnn/acl_meta.h"

#ifdef __cplusplus
extern "C" {
#endif

/* funtion: aclnnLaserAttentionGetWorkspaceSize
 * parameters :
 * query : required
 * key : required
 * value : required
 * attenMaskOptional : optional
 * alibiMaskOptional : optional
 * dropMaskOptional : optional
 * scaleValue : required
 * headNum : required
 * inputLayout : required
 * keepProb : optional
 * preTokens : optional
 * nextTokens : optional
 * isHighPrecision : optional
 * softmaxLogMaxSumOut : required
 * attentionOutOut : required
 * workspaceSize : size of workspace(output).
 * executor : executor context(output).
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLaserAttentionGetWorkspaceSize(
    const aclTensor *query,
    const aclTensor *key,
    const aclTensor *value,
    const aclTensor *attenMaskOptional,
    const aclTensor *alibiMaskOptional,
    const aclTensor *dropMaskOptional,
    double scaleValue,
    int64_t headNum,
    char *inputLayout,
    double keepProb,
    int64_t preTokens,
    int64_t nextTokens,
    bool isHighPrecision,
    const aclTensor *softmaxLogMaxSumOut,
    const aclTensor *attentionOutOut,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

/* funtion: aclnnLaserAttention
 * parameters :
 * workspace : workspace memory addr(input).
 * workspaceSize : size of workspace(input).
 * executor : executor context(input).
 * stream : acl stream.
 */
__attribute__((visibility("default")))
aclnnStatus aclnnLaserAttention(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);

#ifdef __cplusplus
}
#endif

#endif
