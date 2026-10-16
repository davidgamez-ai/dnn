import { Hyperparameters } from "./Hyperparameters.js";
import { Network } from "./Network.js";
import { Trainer } from "./Trainer.js";
import { Tester } from "./Tester.js";
import { XORTester } from "./XORTester.js";

const hyperparameters: Hyperparameters = new Hyperparameters();
hyperparameters.randomSeed = 9;
hyperparameters.epochs = 5;

const xorTester = new XORTester(hyperparameters);
xorTester.run();
