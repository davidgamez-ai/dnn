import { Hyperparameters } from "./Hyperparameters.js";
import { Trainer } from "./Trainer.js";

const hyperparameters: Hyperparameters = new Hyperparameters();
hyperparameters.randomSeed = 123;


const trainer:Trainer = new Trainer(hyperparameters);
trainer.loadData();

trainer.train();


