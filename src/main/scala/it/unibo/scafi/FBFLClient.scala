package it.unibo.scafi

import it.unibo.alchemist.model.scafi.ScafiIncarnationForAlchemist._
import it.unibo.interop.PythonModules.flUtils
import it.unibo.learning.model.Dataset
import it.unibo.scafi.Utils._
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}

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
  private lazy val trainingData = flUtils.to_subset(data.dataset, trainingDataIndexes.toPythonProxy)
  private lazy val validationData = flUtils.to_subset(data.dataset, validationDataIndexes.toPythonProxy)
  private lazy val epochs = sense[Int](Molecules.epochs)
  private lazy val batchSize = sense[Int](Molecules.batchSize)
  private lazy val experiment = sense[String](Molecules.experiment)
  private val radius = 4.5
  private val impulsesEvery = 5

  private type NeuralNetwork = py.Dynamic

  override def main(): Unit = {

    rep((init, 0)){ case (localModel, t) =>

      val aggregators = S(radius, nbrRange)

      // Local Training
      val tick = t + 1
      val trainingResult = flUtils.training(localModel, trainingData, epochs, batchSize, experiment)
      val evolvedModel = py"$trainingResult[0]"
      val trainingLoss = py"$trainingResult[1]".as[Double]
      val validationResults = flUtils.evaluate(evolvedModel, validationData, batchSize, experiment)
      val validationLoss = py"$validationResults[0]".as[Double]
      val validationAccuracy = py"$validationResults[1]".as[Double]

      // Logging
      node.put(Molecules.trainingLoss, trainingLoss)
      node.put(Molecules.validationLoss, validationLoss)
      node.put(Molecules.validationAccuracy, validationAccuracy)
      node.put(Molecules.isAggregator, aggregators)
      node.put(Molecules.localModel, localModel)
      node.put(Molecules.areaId, data.areaId)
      node.put("Training set size", flUtils.dataset_stats(trainingData).as[Double])


      // SCR Updates
      val potential = classicGradient(aggregators)
      val info = C[Double, Set[NeuralNetwork]](potential, _ ++ _, Set(evolvedModel), Set.empty)
      val aggregatedModel = aggregate(info)
      val sharedModel = broadcast(aggregators, aggregatedModel)
      mux(tick % impulsesEvery == 0) { (aggregate(Set(sharedModel, evolvedModel)), tick) } { (evolvedModel, tick) }
    }

  }

  private def init: NeuralNetwork = sense[NeuralNetwork](Molecules.globalModel)

  private def aggregate(models: Set[NeuralNetwork]): NeuralNetwork =
    flUtils.average_weights(models.toList.toPythonProxy, List.fill(models.size)(1.0).toPythonProxy)


}
