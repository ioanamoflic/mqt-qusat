#include "CircuitOptimizer.hpp"
#include "SatEncoder.hpp"
#include "algorithms/RandomCliffordCircuit.hpp"

#include <ctime>
#ifdef _MSC_VER
#define localtime_r(a, b) (localtime_s(b, a) == 0 ? b : NULL)
#endif

#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>
#include <locale>
#include "omp.h"

/* Benchmarking */
std::vector<std::string> getAllCompBasisStates(std::size_t nrQubits) {
  if (nrQubits == 1) {
    return {"I", "Z"};
  }
  std::vector<std::string> rest = getAllCompBasisStates(nrQubits - 1);
  std::vector<std::string> appended;
  for (const auto& s : rest) {
    appended.push_back(s + 'I');
    appended.push_back(s + 'Z');
  }
  return appended;
}

std::string opToString(qc::OpType type) {
  switch (type) {
    case qc::X: return "X";
    case qc::Y: return "Y";
    case qc::Z: return "Z";
    case qc::S: return "S";
    case qc::H: return "H";
    default: return "err";
  }
}

void writeCirqFile(std::string benchmarkFilesPath, std::size_t qubitCnt, std::size_t depth, qc::RandomCliffordCircuit circOne)
{
  std::ofstream outfileCirq(benchmarkFilesPath + "Ioana-EC-" +
                            std::to_string(qubitCnt) + "-" + std::to_string(depth)  +
                            ".txt");
  outfileCirq << "[";
  std::string cirqOperation;
  for (auto & op : circOne)
  {
    std::cout << "Op.type:" << op->getType() << std::endl;
    if (op->getType() == qc::X && op->getNcontrols() > 0)
    {
      const auto control = op->getControls().begin()->qubit;
      const auto target = op->getTargets()[0];
      cirqOperation = "cirq.CX.on(q" + std::to_string(control) + ", q" + std::to_string(target) + "), ";
    }
    else
    {
      const auto q = op->getTargets()[0];
      if (op->getType() == qc::Sdg) {
        cirqOperation = "cirq.S.on(q" + std::to_string(q) + ")**-1, ";
      } else {
        cirqOperation = "cirq." + opToString(op->getType()) + ".on(q" + std::to_string(q) + "), ";
      }
    }
    outfileCirq << cirqOperation;
  }
  outfileCirq << "]";
  outfileCirq.close();
}

class SatEncoderBenchmarking : public testing::TestWithParam<std::string> {
public:
  const std::string benchmarkFilesPath;
};

TEST_F(SatEncoderBenchmarking,
       EquivalenceCheckingGrowingNrOfQubits) { // Equivalence Checking
  try {
    // Paper Evaluation:
    // const std::size_t  depth         = 1000;
    const std::size_t depth    = 100;
    std::size_t       qubitCnt = 4;
    const std::size_t stepsize = 4;
    // Paper Evaluation:
    // const std::size_t  maxNrOfQubits = 128;
    const std::size_t  maxNrOfQubits = 16;
    std::random_device rd;
    std::random_device rd2;
    std::random_device rd3;
    std::random_device rd4;
    std::ostringstream oss;
    std::mt19937       gen(rd());
    std::mt19937       gen2(rd());
    auto               t = std::time(nullptr);
    struct tm          now{};
    localtime_r(&t, &now);
    oss << std::put_time(&now, "%d-%m-%Y");
    auto timestamp = oss.str();

    std::ofstream outfile(benchmarkFilesPath + "EC-" + timestamp + ".json");
    outfile << "{ \"benchmarks\" : [";

    auto                                       ipts = getAllCompBasisStates(5);
    std::uniform_int_distribution<std::size_t> distr(0U, 31U);

    #pragma omp parallel for
    for (qubitCnt=4; qubitCnt < maxNrOfQubits; qubitCnt += stepsize) {
          const auto thread_id = omp_get_thread_num();
          const auto num_threads = omp_get_num_threads();
          std::cout << "Thread " << thread_id << " of " << num_threads << " is running iteration " << qubitCnt << std::endl;

          SatEncoder satEncoder;
          std::vector<std::string> inputs;
          for (size_t j = 0; j < 18; j++) {
            inputs.emplace_back(ipts.at(distr(gen2)));
          }
          qc::RandomCliffordCircuit circOne(qubitCnt, depth, gen());
          qc::CircuitOptimizer::flattenOperations(circOne);
          auto circTwo = circOne;
          if (qubitCnt != 4) {
            outfile << ", ";
          }

          satEncoder.testEqual(circOne, circTwo, inputs); // equivalent case
          outfile << satEncoder.to_json().dump(2U);

          std::cout << "Tested for: " << qubitCnt << std::endl;
          writeCirqFile(benchmarkFilesPath, qubitCnt, depth, circOne);
        }

    qubitCnt = 4;
    for (; qubitCnt < maxNrOfQubits; qubitCnt += stepsize) {
      std::cout << "Nr Qubits: " << qubitCnt << std::endl;
      SatEncoder               satEncoder;
      std::vector<std::string> inputs;
      for (size_t k = 0; k < 18; k++) {
        inputs.emplace_back(ipts.at(distr(gen2)));
      }

      bool result;
      do {
        SatEncoder                satEncoder1;
        qc::RandomCliffordCircuit circThree(qubitCnt, depth, gen());
        qc::CircuitOptimizer::flattenOperations(circThree);
        auto                                       circFour = circThree;
        std::uniform_int_distribution<std::size_t> distr2(
            0U, circFour.size()); // random error location in circuit
        circFour.erase(circFour.begin() + static_cast<int>(distr2(gen)));
        outfile << ", ";
        result = satEncoder1.testEqual(circThree, circFour,
                                       inputs); // non-equivalent case
        outfile << satEncoder1.to_json().dump(2U);
      } while (result);
    }
    outfile << "]}";
    outfile.close();
  } catch (std::exception& e) {
    std::cerr << "EXCEPTION THROWN" << std::endl;
    std::cout << e.what() << std::endl;
  }
}