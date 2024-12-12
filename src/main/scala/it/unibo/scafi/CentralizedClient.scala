package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.interop.PythonModules.flUtils
import it.unibo.learning.model.Dataset
import it.unibo.scafi.Utils.slices
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

class CentralizedClient extends
  AggregateProgram
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
  private lazy val trainingData = flUtils.to_subset(data.dataset, trainingDataIndexes.toPythonProxy)
  private lazy val validationData = flUtils.to_subset(data.dataset, validationDataIndexes.toPythonProxy)
  private lazy val epochs = sense[Int](Molecules.epochs)
  private lazy val batchSize = sense[Int](Molecules.batchSize)
  private lazy val experiment = sense[String](Molecules.experiment)

  override def main(): Any = {

    val globalModel = sense[py.Dynamic](Molecules.globalModel)

    val trainingResult = flUtils.training(globalModel, trainingData, epochs, batchSize, experiment)
    val localModel = py"$trainingResult[0]"
    val trainingLoss = py"$trainingResult[1]".as[Double]
    val validationResults = flUtils.evaluate(globalModel, validationData, batchSize)
    val validationLoss = py"$validationResults[0]".as[Double]
    val validationAccuracy = py"$validationResults[1]".as[Double]

    node.put(Molecules.trainingLoss, trainingLoss)
    node.put(Molecules.validationLoss, validationLoss)
    node.put(Molecules.validationAccuracy, validationAccuracy)

    node.put(Molecules.localModel, localModel)
  }

}
