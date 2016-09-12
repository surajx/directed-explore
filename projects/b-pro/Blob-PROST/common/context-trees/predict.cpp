/******************************
      Author: Joel Veness
        Date: 2011-2013
******************************/

#include <iostream>

#include "ctw_mod.hpp"

// boost includes
#include <boost/program_options.hpp>
#include <boost/progress.hpp>
#include <boost/filesystem/operations.hpp>

namespace po = boost::program_options;

// bit history data structure
history_t history;

/* We initialize the context with zeros for convenience. */
void zeroFill(history_t& h, size_t n) {
  for (size_t i = 0; i < n; ++i)
    h.push_back(0);
}

/* create the file compressor, or die if creation fails */
static Compressor* buildCompressor(unsigned int d, std::string method) {
  // ensure history always has enough context
  history.clear();
  zeroFill(history, d);

  Compressor* c = NULL;

  if (method == "ctw") {
    c = new ContextTree(history, d);
  }

  if (c == NULL) {
    std::cout << "Unknown compression method." << std::endl;
    std::exit(1);
  }

  return c;
}

/* application entry point */
int main(int argc, char* argv[]) {
  try {
    int d = atoi(argv[1]);
    std::string method(argv[2]);
    Compressor* c = buildCompressor(3, method);
    int bit_string[50] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
                          1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0,
                          1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0};
    for (int i = 0; i < sizeof(bit_string) / sizeof(int); i++) {
      double p = c->prob(1);
      double before = c->logBlockProbability();
      c->update(bit_string[i]);
      double after = c->logBlockProbability();
      assert(after < before);
      assert(p > 0.0 && p < 1.0);
    }

  } catch (std::exception& e) {
    std::cout << "error: " << e.what() << std::endl;
    return 1;
  }

  return 0;
}
