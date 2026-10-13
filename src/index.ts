import { Hyperparameters } from "./Hyperparameters.js";
import { Network } from "./Network.js";
import { Trainer } from "./Trainer.js";
import { Tester } from "./Tester.js";

const hyperparameters: Hyperparameters = new Hyperparameters();
hyperparameters.randomSeed = 123;

const tester:Tester = new Tester(hyperparameters);
tester.run();


