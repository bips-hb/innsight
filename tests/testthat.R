library(testthat)
library(innsight)

if (Sys.getenv("TORCH_TEST", unset = 0) == 1) {
  set.seed(42)
  torch::torch_manual_seed(42)
  tensorflow::set_random_seed(43)

  test_check("innsight")
}
