import { Hyperparameters } from "./Hyperparameters.js";
import { Trainer } from "./Trainer.js";
const hyperparameters = new Hyperparameters();
hyperparameters.randomSeed = 123;
const trainer = new Trainer(hyperparameters);
trainer.loadData();
trainer.train();
//# sourceMappingURL=index.js.map