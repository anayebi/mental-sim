from ptutils.model_training.runner import Runner


class VideoRunner(Runner):
    def train(self, config_file):
        if config_file["trainer"] == "FitVidTrainer":
            from mpmodels.model_training.fitvid_trainer import FitVidTrainer

            print("Using FitVidTrainer")
            trainer = FitVidTrainer(config_file)
        elif config_file["trainer"] == "SVGTrainer":
            from mpmodels.model_training.svg_trainer import SVGTrainer

            print("Using SVGTrainer")
            trainer = SVGTrainer(config_file)
        elif config_file["trainer"] == "SelfSupLossMinTrainer":
            from mpmodels.model_training.lossmin_trainer import SelfSupLossMinTrainer

            print("Using SelfSupLossMinTrainer")
            trainer = SelfSupLossMinTrainer(config_file)
        else:
            raise ValueError("Invalid task.")

        trainer.train()


if __name__ == "__main__":
    ## uncomment if you want to debug
    # import os
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="", required=True)
    parser.add_argument("--resume-epoch", type=str, default=None)
    args = parser.parse_args()
    runner = VideoRunner()
    runner.run(args)
