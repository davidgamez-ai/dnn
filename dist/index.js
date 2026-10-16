import { Hyperparameters } from "./Hyperparameters.js";
import { XORTester } from "./XORTester.js";
const hyperparameters = new Hyperparameters();
hyperparameters.randomSeed = 9;
hyperparameters.epochs = 5;
const xorTester = new XORTester(hyperparameters);
xorTester.run();
//# sourceMappingURL=index.js.map