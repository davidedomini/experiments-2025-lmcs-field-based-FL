package it.unibo.alchemist.model.layers

import it.unibo.alchemist.model.{Environment, Layer, Position}
import it.unibo.learning.model.Dataset

class PhenomenaDistribution[P <: Position[P]](
  environment: Environment[_, P],
  private val xStart: Double,
  private val yStart: Double,
  private val xEnd: Double,
  private val yEnd: Double,
  val areas: Int
) extends Layer[Dataset, P] {

  private lazy val subareas: List[(P, P)] = computeSubAreas(
    environment.makePosition(xStart, yStart),
    environment.makePosition(xEnd, yEnd)
  )

  lazy val dataByPosition: Map[P, Int] = subareas
    .zipWithIndex
    .map { case (p, index) => (center(p), index) }
    .toMap

  override def getValue(p: P): Dataset = {
    val id = dataByPosition
      .map { case (position, id) => position.distanceTo(p) -> id }
      .minBy { case (distance, _) => distance }
      ._2
   Dataset(id, null, null)
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

}
