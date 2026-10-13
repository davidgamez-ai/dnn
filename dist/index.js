import { Hyperparameters } from "./Hyperparameters.js";
import { Tester } from "./Tester.js";
const hyperparameters = new Hyperparameters();
hyperparameters.randomSeed = 123;
const tester = new Tester(hyperparameters);
tester.run();
//# sourceMappingURL=index.js.map