import typing
import torch
import pydantic


class WakeWordConfig(pydantic.BaseModel):
  input_size: int
  hidden_size: int
  output_size: int


class DataLoaderConfig(pydantic.BaseModel):
  data: typing.Any
  batch_size: int
  num_workers: int


class OptimizerConfig(pydantic.BaseModel):
  lr: float


class CriterionConfig(pydantic.BaseModel):
  loss_function: typing.Any

  @pydantic.field_validator("loss_function")
  def validate_loss_function(cls, value):
    if hasattr(torch.nn, value):
      loss_class = getattr(torch.nn, value)
      return loss_class
    else:
      raise ValueError(f"{value} is not a valid torch.nn loss class")


class ModelInitializerConfig(pydantic.BaseModel):
  wakeword_config: WakeWordConfig
  dataloader_config: DataLoaderConfig
  optimizer_config: OptimizerConfig
  criterion_config: CriterionConfig


if __name__ == "__main__":
  MIC = ModelInitializerConfig(wakeword_config=WakeWordConfig(input_size=1, output_size=1, hidden_size=1),
                               dataloader_config=DataLoaderConfig(data={}, batch_size=10, num_workers=1),
                               optimizer_config=OptimizerConfig(lr=0.001),
                               criterion_config=CriterionConfig(loss_function="BCELoss"))
