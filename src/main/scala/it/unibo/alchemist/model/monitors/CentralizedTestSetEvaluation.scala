package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.model.{Environment, Position, Time}

import scala.jdk.CollectionConverters.IteratorHasAsScala
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.exporter.TestDataExporter
import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.interop.PythonModules.flUtils
import me.shadaj.scalapy.py.PyQuote
import it.unibo.scafi.Molecules
import me.shadaj.scalapy.py

class CentralizedTestSetEvaluation[P <: Position[P]](
  batchSize: Int,
  experiment: String,
  areas: Int,
  seed: Int
) extends OutputMonitor[Any, P] {
  override def finished(environment: Environment[Any, P], time: Time, step: Long): Unit = {
    val model = environment
      .getNodes
      .iterator()
      .asScala
      .toList
      .head
      .getConcentration(new SimpleMolecule(Molecules.globalModel))
      .asInstanceOf[py.Dynamic]
    val dataset = flUtils.get_dataset(experiment, false)
    val results = flUtils.evaluate(model, dataset, batchSize, experiment)
    val testAccuracy = py"$results[1]".as[Double]
    TestDataExporter.CSVExport(
      List(testAccuracy),
      s"data/test-centralized-FL_experiment-${experiment}_seed-${seed}_areas-$areas"
    )
  }
}
