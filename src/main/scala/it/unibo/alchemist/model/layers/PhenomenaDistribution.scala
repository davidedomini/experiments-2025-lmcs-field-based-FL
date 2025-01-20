package it.unibo.alchemist.model.layers

import it.unibo.learning.model.{Dataset, Dirichlet, Hard, IID, Partitioning}
import it.unibo.alchemist.model.{Environment, Layer, Position}
import it.unibo.interop.PythonModules._
import me.shadaj.scalapy.py

import scala.util.Random

class PhenomenaDistribution[P <: Position[P]](
  environment: Environment[_, P],
  private val xStart: Double,
  private val yStart: Double,
  private val xEnd: Double,
  private val yEnd: Double,
  val areas: Int,
  val partitioning: Partitioning,
  val datasetName: String,
  val trainFraction: Double,
  val seed: Int
) extends Layer[Dataset, P] {

  private val random = new Random(seed)

  private val dataset = getDataset(datasetName)

  private lazy val classes: Int = dataset.classes.as[List[String]].size

  private lazy val subareas: List[(P, P)] = computeSubAreas(
    environment.makePosition(xStart, yStart),
    environment.makePosition(xEnd, yEnd)
  )

  private val dataMapping = partitions.as[Map[Int, List[Int]]]

  private lazy val subsets = dataMapping
    .map { case (id, indexes) =>
      val trainSize = math.floor(indexes.size * trainFraction).toInt
      val shuffledIndexes = random.shuffle(indexes)
      val trainIndexes = shuffledIndexes.take(trainSize)
      val validationIndexes = shuffledIndexes.takeRight(shuffledIndexes.size - trainSize)
      id -> (trainIndexes, validationIndexes)
//      val d = flUtils.get_subset(dataset, indexes.toPythonProxy)
//      id -> (py"$d[0]", py"$d[1]") // id -> train_data, validation_data
    }

  private lazy val idByPosition: Map[P, Int] = subareas
    .zipWithIndex
    .map { case (p, index) => (center(p), index) }
    .toMap

  override def getValue(p: P): Dataset = {
    val id = getAreaIdByPosition(p)
    val data = subsets.getOrElse(id, throw new IllegalStateException(s"Data for area $id not found"))
   Dataset(id, data._1, data._2, dataset)
  }

  def getAreaIdByPosition(p: P): Int = {
    idByPosition
      .map { case (position, id) => position.distanceTo(p) -> id}
      .minBy { case (distance, _) => distance }
      ._2
  }

  private def computeSubAreas(start: P, end: P): List[(P, P)] = {
    val rows = math.sqrt(areas).toInt
    val cols = (areas + rows - 1) / rows
    val width = math.abs(end.getCoordinate(0) - start.getCoordinate(0))
    val height = math.abs(end.getCoordinate(1) - start.getCoordinate(1))
    val rowHeight = height / rows
    val colWidth = width / cols
    val result = for {
      row <- 0 until rows
      col <- 0 until cols
    } yield {
      val x1 = start.getCoordinate(0) + col * colWidth
      val y1 = start.getCoordinate(1) + row * rowHeight
      val x2 = x1 + colWidth
      val y2 = y1 + rowHeight
      environment.makePosition(x1, y1) -> environment.makePosition(x2, y2)
    }
    result.toList
  }

  private def center(p: (P, P)): P = {
    val xCenter = (p._1.getCoordinate(0) + p._2.getCoordinate(0)) / 2
    val yCenter = (p._1.getCoordinate(1) + p._2.getCoordinate(1)) / 2
    environment.makePosition(xCenter, yCenter)
  }

  private def partitions: py.Dynamic = partitioning match {
    case IID =>
      val mapping = flUtils.iid_mapping(areas, classes)
      flUtils.partioniong(mapping, dataset)
    case Hard =>
      val mapping = flUtils.hard_non_iid_mapping(areas, classes)
      flUtils.partioniong(mapping, dataset)
    case Dirichlet(beta) =>
      flUtils.dirichlet_partitioning(dataset, areas, beta)
  }

  private def getDataset(name: String): py.Dynamic =
    flUtils.get_dataset(name)

}