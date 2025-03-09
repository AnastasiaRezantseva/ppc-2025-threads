
#include <gtest/gtest.h>

#include <random>

#include "seq/rezantseva_shell_batcher_seq/include/rezantseva_shell_batcher_seq.hpp"
static int offset = 0;

namespace rezantseva_shell_batcher_seq {

std::vector<double> createRandomVector(const int vec_size) {
  std::vector<double> random_vec(vec_size);
  std::mt19937 gen;
  gen.seed((unsigned)time(nullptr) + ++offset);
  std::uniform_real_distribution<double> dist(-50.0, 50.0);

  for (int i = 0; i < vec_size; i++) {
    random_vec[i] = dist(gen);
  }

  return random_vec;
}
}  // namespace rezantseva_shell_batcher_seq

TEST(rezantseva_shell_batcher_seq, shell_sort_test) {
  // Create data
  std::vector<double> input = {34.27, -2.9, 45.37, 12.7, 8.02, -10.1, 3.4, 0.0, 7.89, 23.3, 34.01};
  std::vector<double> sorted = {-10.1, -2.9, 0.0, 3.4, 7.89, 8.02, 12.7, 23.3, 34.01, 34.27, 45.37};
  std::vector<double> out(input.size(), 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_EQ(test_shell_batcher.Validation(), true);
  test_shell_batcher.PreProcessing();
  test_shell_batcher.Run();
  test_shell_batcher.PostProcessing();

  EXPECT_EQ(sorted, out);
}

TEST(rezantseva_shell_batcher_seq, shell_sort_test_empty_vec) {
  // Create data
  std::vector<double> input = {};
  std::vector<double> out(1, 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_FALSE(test_shell_batcher.Validation());
}

TEST(rezantseva_shell_batcher_seq, shell_sort_test_vec_size_1) {
  // Create data
  std::vector<double> input = {1.9};
  std::vector<double> out(1, 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_FALSE(test_shell_batcher.Validation());
}

TEST(rezantseva_shell_batcher_seq, shell_sort_test_with_repeating_elements) {
  // Create data
  std::vector<double> input = {34.27, -2.9, 45.37, 12.7, 8.02, -10.1, 45.37, 3.4, 0.0, 7.89, 23.3, 34.01};
  std::vector<double> sorted = {-10.1, -2.9, 0.0, 3.4, 7.89, 8.02, 12.7, 23.3, 34.01, 34.27, 45.37, 45.37};
  std::vector<double> out(input.size(), 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_EQ(test_shell_batcher.Validation(), true);
  test_shell_batcher.PreProcessing();
  test_shell_batcher.Run();
  test_shell_batcher.PostProcessing();

  EXPECT_EQ(sorted, out);
}

TEST(rezantseva_shell_batcher_seq, shell_sort_test_with_random_vector_size_50) {
  // Create data
  int vec_size = 50;
  std::vector<double> input = rezantseva_shell_batcher_seq::createRandomVector(vec_size);
  std::vector<double> out(input.size(), 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_EQ(test_shell_batcher.Validation(), true);
  test_shell_batcher.PreProcessing();
  test_shell_batcher.Run();
  test_shell_batcher.PostProcessing();

  std::sort(input.begin(), input.end());
  EXPECT_EQ(input, out);
}

TEST(rezantseva_shell_batcher_seq, shell_sort_test_with_random_vector_size_3) {
  // Create data
  int vec_size = 3;
  std::vector<double> input = rezantseva_shell_batcher_seq::createRandomVector(vec_size);
  std::vector<double> out(input.size(), 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_EQ(test_shell_batcher.Validation(), true);
  test_shell_batcher.PreProcessing();
  test_shell_batcher.Run();
  test_shell_batcher.PostProcessing();

  std::sort(input.begin(), input.end());
  EXPECT_EQ(input, out);
}

TEST(rezantseva_shell_batcher_seq, shell_sort_test_with_random_vector_size_25) {
  // Create data
  int vec_size = 25;
  std::vector<double> input = rezantseva_shell_batcher_seq::createRandomVector(vec_size);
  std::vector<double> out(input.size(), 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_EQ(test_shell_batcher.Validation(), true);
  test_shell_batcher.PreProcessing();
  test_shell_batcher.Run();
  test_shell_batcher.PostProcessing();

  std::sort(input.begin(), input.end());
  EXPECT_EQ(input, out);
}

TEST(rezantseva_shell_batcher_seq, shell_sort_test_with_random_vector_size_20) {
  // Create data
  int vec_size = 20;
  std::vector<double> input = rezantseva_shell_batcher_seq::createRandomVector(vec_size);
  std::vector<double> out(input.size(), 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_EQ(test_shell_batcher.Validation(), true);
  test_shell_batcher.PreProcessing();
  test_shell_batcher.Run();
  test_shell_batcher.PostProcessing();

  std::sort(input.begin(), input.end());
  EXPECT_EQ(input, out);
}

TEST(rezantseva_shell_batcher_seq, shell_sort_test_with_random_vector_size_100) {
  // Create data
  int vec_size = 100;
  std::vector<double> input = rezantseva_shell_batcher_seq::createRandomVector(vec_size);
  std::vector<double> out(input.size(), 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_EQ(test_shell_batcher.Validation(), true);
  test_shell_batcher.PreProcessing();
  test_shell_batcher.Run();
  test_shell_batcher.PostProcessing();

  std::sort(input.begin(), input.end());
  EXPECT_EQ(input, out);
}

TEST(rezantseva_shell_batcher_seq, shell_sort_test_with_random_vector_size_1000) {
  // Create data
  int vec_size = 1000;
  std::vector<double> input = rezantseva_shell_batcher_seq::createRandomVector(vec_size);
  std::vector<double> out(input.size(), 0.0);
  // Create task_data
  auto task_data_seq = std::make_shared<ppc::core::TaskData>();

  task_data_seq->inputs.emplace_back(reinterpret_cast<uint8_t *>(input.data()));
  task_data_seq->inputs_count.emplace_back(input.size());

  task_data_seq->outputs.emplace_back(reinterpret_cast<uint8_t *>(out.data()));
  task_data_seq->outputs_count.emplace_back(out.size());

  // Create Task
  rezantseva_shell_batcher_seq::ShellBatcherSortSequential test_shell_batcher(task_data_seq);
  ASSERT_EQ(test_shell_batcher.Validation(), true);
  test_shell_batcher.PreProcessing();
  test_shell_batcher.Run();
  test_shell_batcher.PostProcessing();

  std::sort(input.begin(), input.end());
  EXPECT_EQ(input, out);
}
