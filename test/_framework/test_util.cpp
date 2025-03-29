/**
 * @file se_test_util.cpp
 * @brief Implementation for sail engine test utility functions
 * @author sailing-innocent
 * @date 2023-09-15
 */

#define DOCTEST_CONFIG_IMPLEMENT
#include "test_util.h"

#include <string_view>
#include <vector>

namespace lcgs::test {

static std::vector<const char*> args;

inline void dt_remove(const char** argv_in) noexcept {
	args.clear();
	for (; *argv_in; ++argv_in) {
		if (!std::string_view{*argv_in}.starts_with("--dt-")) {
			args.emplace_back(*argv_in);
		}
	}
	args.emplace_back(nullptr);
}

int argc() noexcept {
	return static_cast<int>(args.size());
}
const char* const* argv() noexcept {
	return args.data();
}

bool float_span_equal(std::span<float> a, std::span<float> b) {
	int N = a.size();
	if (N != b.size()) {
		return false;
	}
	for (int i = 0; i < N; ++i) {
		if (a[i] != doctest::Approx(b[i])) {
			return false;
		}
	}
	return true;
}

}// namespace lcgs::test

int main(int argc, const char** argv) {
	doctest::Context context(argc, argv);
	lcgs::test::dt_remove(argv);
	auto test_result = context.run();
	if (context.shouldExit()) {
		return test_result;
	}
	return test_result;
}