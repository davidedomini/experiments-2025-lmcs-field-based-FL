package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.learning.model.Dataset
import it.unibo.scafi.Utils._

class FBFLClient
  extends AggregateProgram
  with BuildingBlocks
  with StandardSensors
  with ScafiAlchemistSupport {

  private lazy val data = senseEnvData[Dataset](Molecules.phenomena)
  private lazy val myIdInArea = sense[Int](Molecules.idInArea)
  private lazy val neighbors = sense[Int](Molecules.neighbors)
  private lazy val trainingDataInterval = slices(data.trainingData.size, neighbors)(myIdInArea)
  private lazy val validationDataInterval = slices(data.validationData.size, neighbors)(myIdInArea)
  private lazy val trainingDataIndexes = data.trainingData.slice(trainingDataInterval._1, trainingDataInterval._2)
  private lazy val validationDataIndexes = data.validationData.slice(validationDataInterval._1, validationDataInterval._2)

  override def main(): Unit = {

    node.put("AreaId", data.areaId)
    node.put("TrainSize", data.trainingData.size)
    node.put("ValSize", data.validationData.size)
    node.put("TrainInterval", trainingDataInterval)
    node.put("ValidationInterval", validationDataInterval)

  }

}
