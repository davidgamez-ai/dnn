import { Hyperparameters } from "./Hyperparameters.js";
import { Network } from "./Network.js";
import { Trainer } from "./Trainer.js";
import { Tester } from "./Tester.js";
import { XORTester } from "./XORTester.js";

// learningRate 0.01, epochs 2000, randomSeed 9; change any of them here.
const hyperparameters: Hyperparameters = XORTester.defaultHyperparameters();

const xorTester = new XORTester(hyperparameters);
xorTester.run();
