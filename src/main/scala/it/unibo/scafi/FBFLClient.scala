package it.unibo.scafi
import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.learning.model.Dataset

class FBFLClient
  extends AggregateProgram
  with BuildingBlocks
  with StandardSensors
  with ScafiAlchemistSupport {

  override def main(): Unit = {
    val data = senseEnvData[Dataset](Molecules.phenomena)
    node.put("AreaId", data.areaId)
    node.put("TrainSize", data.trainingData.size)
    node.put("ValSize", data.validationData.size)
  }

}
