package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.model.{Environment, Node, Position, Time}
import it.unibo.learning.model.{Dirichlet, Hard, IID, Partitioning}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import scala.jdk.CollectionConverters.IteratorHasAsScala
import me.shadaj.scalapy.py.{PyQuote, SeqConverters}
import it.unibo.alchemist.exporter.TestDataExporter
import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.interop.PythonModules.flUtils
import it.unibo.scafi.Molecules
import me.shadaj.scalapy.py

class DecentralizedTestSetEvaluation [P <: Position[P]](
 batchSize: Int,
 experiment: String,
 areas: Int,
 seed: Int,
 partitioning: Partitioning
) extends OutputMonitor[Any, P]{

  private val dataset = flUtils.get_dataset(experiment, false)
  private val classes = dataset.classes.as[List[String]].size
  private val dataMapping = partitions.as[Map[Int, List[Int]]]

  override def finished(environment: Environment[Any, P], time: Time, step: Long): Unit = {
    val leaders = findLeaders(environment)
    val testAccuracies = leaders
      .map { l =>
        val areaID = l.getConcentration(new SimpleMolecule(Molecules.areaId)).asInstanceOf[Int]
        val model = l.getConcentration(new SimpleMolecule(Molecules.localModel)).asInstanceOf[py.Dynamic]
        (areaID, model)
      }
      .map { case (areaID, model) =>
        val indexes = dataMapping.getOrElse(areaID, throw new IllegalStateException(s"Data for area id $areaID not found"))
        val subset = flUtils.to_subset(dataset, indexes.toPythonProxy)
        flUtils.evaluate(model, subset, batchSize, experiment)
      }
      .map { result => py"$result[1]".as[Double] }
    TestDataExporter.CSVExport(
      testAccuracies,
      s"data/test-FBFL-${experiment}_seed-${seed}_areas-${areas}_partitioning-${getPartitioning(partitioning)}"
    )
  }

  private def getPartitioning(partitioning: Partitioning):String = partitioning match {
    case IID => "IID"
    case Hard => "Hard"
    case Dirichlet(beta) => s"Dirichlet-$beta"
  }

  private def findLeaders(environment: Environment[Any, P]): List[Node[Any]] =
    environment
    .getNodes
    .iterator()
    .asScala
    .toList
    .filter(_.getConcentration(new SimpleMolecule(Molecules.isAggregator)).asInstanceOf[Boolean])

  private def partitions: py.Dynamic = partitioning match {
    case IID =>
      val mapping = flUtils.iid_mapping(areas, classes)
      flUtils.partitioning(mapping, dataset)
    case Hard =>
      val mapping = flUtils.hard_non_iid_mapping(areas, classes)
      flUtils.partitioning(mapping, dataset)
    case Dirichlet(beta) =>
      flUtils.dirichlet_partitioning(dataset, areas, beta)
  }

}
