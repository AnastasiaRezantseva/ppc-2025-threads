#include "omp/rezantseva_shell_batcher_omp/include/rezantseva_shell_batcher_omp.hpp"

#include <iostream>
#include <vector>
size_t threads = 4;

std::vector<double> rezantseva_shell_batcher_omp::ShellSortSeq(const std::vector<double> &v) {
  size_t n = v.size();
  std::vector<double> result(v);

  for (size_t step = n / 2; step > 0; step /= 2) {
    for (size_t i = step; i < n; i++) {
      double temp = result[i];
      size_t j;
      for (j = i; j >= step && result[j - step] > temp; j -= step) {
        result[j] = result[j - step];
      }
      result[j] = temp;
    }
  }

  return result;
}

void rezantseva_shell_batcher_omp::ShellSortInPlace(std::vector<double> &v) {
  size_t n = v.size();
  for (size_t step = n / 2; step > 0; step /= 2) {
    for (size_t i = step; i < n; i++) {
      double temp = v[i];
      size_t j;
      for (j = i; j >= step && v[j - step] > temp; j -= step) {
        v[j] = v[j - step];
      }
      v[j] = temp;
    }
  }
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortSequential::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] > 1 && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);  // get data
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);        // fill

  unsigned int output_size = task_data->outputs_count[0];
  result_ = std::vector<double>(output_size, 0);
  return true;
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortSequential::RunImpl() {
  result_ = ShellSortSeq(input_);
  return true;
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortSequential::PostProcessingImpl() {
  std::copy(result_.begin(), result_.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
//------------------------------------------------------------------------------------------------------------------//

bool rezantseva_shell_batcher_omp::ShellBatcherSortOMP::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] > 1 && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortOMP::PreProcessingImpl() {  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);  // get data
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);        // fill

  unsigned int output_size = task_data->outputs_count[0];
  result_ = std::vector<double>(output_size, 0);
  return true;
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortOMP::RunImpl() {
  result_ = ShellSortBatcherMergeOMP(input_);
  return true;
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortOMP::PostProcessingImpl() {
  std::copy(result_.begin(), result_.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}

//------------------------------------------------------------------------------------------------------------------//
std::vector<double> rezantseva_shell_batcher_omp::ShellBatcherSortOMP::ShellSortOMP(const std::vector<double> &v) {
  omp_set_num_threads(threads);
  std::vector<double> result(v);
  int v_size = result.size();
  std::size_t delta = v_size / threads;
  std::size_t remainder = v_size % threads;
  std::vector<std::vector<double>> sorted_parts(threads);

#pragma omp parallel shared(result, remainder, sorted_parts)
  {
    std::size_t curr_thread = omp_get_thread_num();

    // Data for current thread
    std::size_t start = curr_thread * delta + (curr_thread < remainder ? curr_thread : remainder);
    std::size_t end = start + delta + (curr_thread < remainder ? 1 : 0);

    std::vector<double> local_vec(result.begin() + start, result.begin() + end);
    ShellSortInPlace(local_vec);
    // Save sorted part

    sorted_parts[curr_thread] = local_vec;
  }

  // Merge sorted parts
  result = sorted_parts[0];
  for (size_t i = 1; i < threads; i++) {
    // result = BatcherMerge(result, sorted_parts[i]);
    // result = MyBatcherMerge(result, sorted_parts[i]);
  }

  return result;
}
//------------------------------------------------------------------------------------------------------------------//

std::vector<double> rezantseva_shell_batcher_omp::ShellBatcherSortOMP::BatcherMerge(std::vector<double> v1,
                                                                                    std::vector<double> v2) {
  std::vector<double> even = EvenBatcher(v1, v2);
  std::vector<double> odd = OddBatcher(v1, v2);
  std::vector<double> result(even.size() + odd.size());
  int even_size = even.size();
  int odd_size = odd.size();
  int i = 0, j = 0, k = 0;

  while (j < even_size && k < odd_size) {
    result[i++] = even[j++];
    result[i++] = odd[k++];
  }

  while (j < even_size) {
    result[i++] = even[j++];
  }

  while (k < odd_size) {
    result[i++] = odd[k++];
  }

#pragma omp parallel for
  for (int h = 0; h < result.size() - 1; h++) {
    if (result[h] > result[h + 1]) {
      std::swap(result[h], result[h + 1]);
    }
  }

  return result;
}

std::vector<double> rezantseva_shell_batcher_omp::ShellBatcherSortOMP::EvenBatcher(std::vector<double> v1,
                                                                                   std::vector<double> v2) {
  int v1_size = v1.size();
  int v2_size = v2.size();
  std::vector<double> result;
  result.reserve(v1_size / 2 + v2_size / 2 + v1_size % 2 + v2_size % 2);

  int i = 0, j = 0;

  while (i < v1_size && j < v2_size) {
    if (v1[i] <= v2[j]) {
      result.push_back(v1[i]);
      i += 2;
    } else {
      result.push_back(v2[j]);
      j += 2;
    }
  }

  while (i < v1_size) {
    result.push_back(v1[i]);
    i += 2;
  }

  while (j < v2_size) {
    result.push_back(v2[j]);
    j += 2;
  }

  return result;
}

std::vector<double> rezantseva_shell_batcher_omp::ShellBatcherSortOMP::OddBatcher(std::vector<double> v1,
                                                                                  std::vector<double> v2) {
  int v1_size = v1.size();
  int v2_size = v2.size();
  std::vector<double> result;
  result.reserve(v1_size / 2 + v2_size / 2);

  int i = 1, j = 1;

  while (i < v1_size && j < v2_size) {
    if (v1[i] <= v2[j]) {
      result.push_back(v1[i]);
      i += 2;
    } else {
      result.push_back(v2[j]);
      j += 2;
    }
  }

  while (i < v1_size) {
    result.push_back(v1[i]);
    i += 2;
  }

  while (j < v2_size) {
    result.push_back(v2[j]);
    j += 2;
  }

  return result;
}

//------------------------------------------------------------------------------------------------------------------//
std::vector<double> rezantseva_shell_batcher_omp::ShellBatcherSortOMP::EvenOddBatcher(std::vector<double> &v1,
                                                                                      std::vector<double> &v2,
                                                                                      bool isEven) {
  int v1_size = v1.size();
  int v2_size = v2.size();
  int start = isEven ? 0 : 1;

  // Calculate result size
  const int result_size = ((v1_size - start + 1) / 2) + ((v2_size - start + 1) / 2);

  std::vector<double> result;
  result.reserve(result_size);

  int i = start, j = start;

  while (i < v1_size && j < v2_size) {
    if (v1[i] <= v2[j]) {
      result.push_back(v1[i]);
      i += 2;
    } else {
      result.push_back(v2[j]);
      j += 2;
    }
  }

  while (i < v1_size) {
    result.push_back(v1[i]);
    i += 2;
  }

  while (j < v2_size) {
    result.push_back(v2[j]);
    j += 2;
  }

  return result;
}

std::vector<double> rezantseva_shell_batcher_omp::ShellBatcherSortOMP::MyBatcherMerge(std::vector<double> &&v1,
                                                                                      std::vector<double> &&v2) {
  std::vector<double> even, odd;
#pragma omp parallel sections
  {
#pragma omp section
    even = EvenOddBatcher(v1, v2, true);
#pragma omp section
    odd = EvenOddBatcher(v1, v2, false);
  }

  std::vector<double> result(even.size() + odd.size());
  int even_size = even.size();
  int odd_size = odd.size();
  int i = 0, j = 0, k = 0;

  while (j < even_size && k < odd_size) {
    result[i++] = even[j++];
    result[i++] = odd[k++];
  }

  while (j < even_size) {
    result[i++] = even[j++];
  }

  while (k < odd_size) {
    result[i++] = odd[k++];
  }

  for (int h = 0; h < result.size() - 1; h++) {
    if (result[h] > result[h + 1]) {
      std::swap(result[h], result[h + 1]);
    }
  }
  return result;
}

std::vector<double> rezantseva_shell_batcher_omp::ShellBatcherSortOMP::ShellSortBatcherMergeOMP(
    const std::vector<double> &v) {
  omp_set_num_threads(threads);

  std::vector<double> result(v);
  size_t v_size = result.size();

  size_t delta = v_size / threads;
  size_t remainder = v_size % threads;
  std::vector<std::vector<double>> sorted_parts(threads);

// Sort parts
#pragma omp parallel shared(result, remainder, sorted_parts)
  {
    size_t curr_thread = omp_get_thread_num();
    size_t start = curr_thread * delta + (curr_thread < remainder ? curr_thread : remainder);
    size_t end = start + delta + (curr_thread < remainder ? 1 : 0);

    std::vector<double> local_vec(result.begin() + start, result.begin() + end);
    ShellSortInPlace(local_vec);
    sorted_parts[curr_thread] = local_vec;
  }

  // Batcher Merge
  while (sorted_parts.size() > 1) {
    const size_t new_size = (sorted_parts.size() + 1) / 2;
    std::vector<std::vector<double>> parts(new_size);
    // how much pairs can merge now
    const size_t pairs_count = sorted_parts.size() / 2;

#pragma omp parallel for
    for (int i = 0; i < pairs_count; ++i) {
      parts[i] = MyBatcherMerge(std::move(sorted_parts[2 * i]), std::move(sorted_parts[2 * i + 1]));
    }

    // Add an unpaired element if there is one
    if (sorted_parts.size() % 2 != 0) {
      parts.back() = std::move(sorted_parts.back());
    }
    // update parts which one need to merge
    sorted_parts = std::move(parts);
  }
  result = sorted_parts[0];
  return result;
}