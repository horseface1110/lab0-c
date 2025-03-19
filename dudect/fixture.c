/** dude, is my code constant time?
 *
 * This file measures the execution time of a given function many times with
 * different inputs and performs a Welch's t-test to determine if the function
 * runs in constant time or not. This is essentially leakage detection, and
 * not a timing attack.
 *
 * Notes:
 *
 *  - the execution time distribution tends to be skewed towards large
 *    timings, leading to a fat right tail. Most executions take little time,
 *    some of them take a lot. We try to speed up the test process by
 *    throwing away those measurements with large cycle count. (For example,
 *    those measurements could correspond to the execution being interrupted
 *    by the OS.) Setting a threshold value for this is not obvious; we just
 *    keep the x% percent fastest timings, and repeat for several values of x.
 *
 *  - the previous observation is highly heuristic. We also keep the uncropped
 *    measurement time and do a t-test on that.
 *
 *  - we also test for unequal variances (second order test), but this is
 *    probably redundant since we're doing as well a t-test on cropped
 *    measurements (non-linear transform)
 *
 *  - as long as any of the different test fails, the code will be deemed
 *    variable time.
 */

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../console.h"
#include "../random.h"

#include "constant.h"
#include "fixture.h"
#include "ttest.h"

#define ENOUGH_MEASURE 10000
#define TEST_TRIES 10
#define DUDECT_NUMBER_PERCENTILES 5

static t_context_t *t;
static t_context_t *t_cropped[DUDECT_NUMBER_PERCENTILES];
static t_context_t *t_second_order;
static int64_t percentiles[DUDECT_NUMBER_PERCENTILES] = {
    0};  // 根據需求初始化百分位門檻數值

/* threshold values for Welch's t-test */
enum {
    t_threshold_bananas = 500, /* Test failed with overwhelming probability */
    t_threshold_moderate = 10, /* Test failed */
};

static void __attribute__((noreturn)) die(void)
{
    exit(111);
}

static void differentiate(int64_t *exec_times,
                          const int64_t *before_ticks,
                          const int64_t *after_ticks)
{
    for (size_t i = 0; i < N_MEASURES; i++)
        exec_times[i] = after_ticks[i] - before_ticks[i];
}

// 新增 prepare_percentiles 函式來根據 exec_times 計算百分位門檻
static int compare_int64(const void *a, const void *b)
{
    int64_t arg1 = *(const int64_t *) a;
    int64_t arg2 = *(const int64_t *) b;
    if (arg1 < arg2)
        return -1;
    else if (arg1 > arg2)
        return 1;
    return 0;
}

static void prepare_percentiles(int64_t *exec_times)
{
    size_t start = DROP_SIZE;
    size_t valid_count = N_MEASURES - 2 * DROP_SIZE;
    if (valid_count == 0)
        return;
    // 直接針對 exec_times 的有效部分排序
    qsort(exec_times + start, valid_count, sizeof(int64_t), compare_int64);
    for (size_t crop_index = 0; crop_index < DUDECT_NUMBER_PERCENTILES;
         crop_index++) {
        size_t pos = start + ((crop_index + 1) * valid_count) /
                                 (DUDECT_NUMBER_PERCENTILES + 1);
        if (pos >= N_MEASURES - DROP_SIZE)
            pos = N_MEASURES - DROP_SIZE - 1;
        percentiles[crop_index] = exec_times[pos];
    }
}

static void update_statistics(const int64_t *exec_times, uint8_t *classes)
{
    // 從第 10 筆開始，捨棄不穩定的初期數據
    for (size_t i = 10; i < N_MEASURES; i++) {
        int64_t difference = exec_times[i];

        // 無效測量（例如 CPU 週期計數器溢出），跳過
        if (difference <= 0)
            continue;

        // 更新原始統計上下文 t
        t_push(t, difference, classes[i]);

        // 根據百分位門檻進行裁剪統計
        for (size_t crop_index = 0; crop_index < DUDECT_NUMBER_PERCENTILES;
             crop_index++) {
            if (difference < percentiles[crop_index])
                t_push(t_cropped[crop_index], difference, classes[i]);
        }

        // 第二階段檢測: 當測量數足夠多時，計算 centered product 的平方並更新
        // t_second_order
        if (t->n[0] > 10000) {
            double centered = (double) difference - t->mean[classes[i]];
            t_push(t_second_order, centered * centered, classes[i]);
        }
    }
}

static bool report(void)  // TODO: report 不需要改
{
    double max_t = fabs(t_compute(t));
    double number_traces_max_t = t->n[0] + t->n[1];
    double max_tau = max_t / sqrt(number_traces_max_t);

    printf("\033[A\033[2K");
    printf("measure: %7.2lf M, ", (number_traces_max_t / 1e6));
    if (number_traces_max_t < ENOUGH_MEASURE) {
        printf("not enough measurements (%.0f still to go).\n",
               ENOUGH_MEASURE - number_traces_max_t);
        return false;
    }

    /* max_t: the t statistic value
     * max_tau: a t value normalized by sqrt(number of measurements).
     *          this way we can compare max_tau taken with different
     *          number of measurements. This is sort of "distance
     *          between distributions", independent of number of
     *          measurements.
     * (5/tau)^2: how many measurements we would need to barely
     *            detect the leak, if present. "barely detect the
     *            leak" = have a t value greater than 5.
     */
    printf("max t: %+7.2f, max tau: %.2e, (5/tau)^2: %.2e.\n", max_t, max_tau,
           (double) (5 * 5) / (double) (max_tau * max_tau));

    /* Definitely not constant time */
    if (max_t > t_threshold_bananas)
        return false;

    /* Probably not constant time. */
    if (max_t > t_threshold_moderate)
        return false;

    /* For the moment, maybe constant time. */
    return true;
}

static bool doit(int mode)
{
    int64_t *before_ticks = calloc(N_MEASURES + 1, sizeof(int64_t));
    int64_t *after_ticks = calloc(N_MEASURES + 1, sizeof(int64_t));
    int64_t *exec_times = calloc(N_MEASURES, sizeof(int64_t));
    uint8_t *classes = calloc(N_MEASURES, sizeof(uint8_t));
    uint8_t *input_data = calloc(N_MEASURES * CHUNK_SIZE, sizeof(uint8_t));

    if (!before_ticks || !after_ticks || !exec_times || !classes ||
        !input_data) {
        die();
    }

    prepare_inputs(input_data, classes);

    bool ret = measure(before_ticks, after_ticks, input_data, mode);
    differentiate(exec_times, before_ticks, after_ticks);
    prepare_percentiles(exec_times);
    update_statistics(exec_times, classes);
    ret &= report();

    free(before_ticks);
    free(after_ticks);
    free(exec_times);
    free(classes);
    free(input_data);

    return ret;
}

static void init_once(void)
{
    init_dut();
    t_init(t);
    for (size_t i = 0; i < DUDECT_NUMBER_PERCENTILES; i++) {
        t_cropped[i] = malloc(sizeof(t_context_t));
        if (!t_cropped[i])
            die();
        t_init(t_cropped[i]);
    }
    t_second_order = malloc(sizeof(t_context_t));
    if (!t_second_order)
        die();
    t_init(t_second_order);
}

static bool test_const(char *text, int mode)
{
    bool result = false;
    t = malloc(sizeof(t_context_t));

    for (int cnt = 0; cnt < TEST_TRIES; ++cnt) {
        printf("Testing %s...(%d/%d)\n\n", text, cnt, TEST_TRIES);
        init_once();
        for (int i = 0; i < ENOUGH_MEASURE / (N_MEASURES - DROP_SIZE * 2) + 1;
             ++i)
            result = doit(mode);
        printf("\033[A\033[2K\033[A\033[2K");
        if (result)
            break;
    }
    free(t);
    return result;
}

#define DUT_FUNC_IMPL(op)                \
    bool is_##op##_const(void)           \
    {                                    \
        return test_const(#op, DUT(op)); \
    }

#define _(x) DUT_FUNC_IMPL(x)
DUT_FUNCS
#undef _
