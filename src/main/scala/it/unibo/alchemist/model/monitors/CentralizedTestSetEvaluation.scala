package it.unibo.alchemist.model.monitors

import it.unibo.alchemist.model.{Environment, Position, Time}

import scala.jdk.CollectionConverters.IteratorHasAsScala
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.boundary.OutputMonitor
import it.unibo.interop.PythonModules.flUtils
import me.shadaj.scalapy.py.PyQuote
import it.unibo.scafi.Molecules

class CentralizedTestSetEvaluation[P <: Position[P]](
  batchSize: Int,
  experiment: String
) extends OutputMonitor[Any, P] {
  override def finished(environment: Environment[Any, P], time: Time, step: Long): Unit = {
    val model = environment
      .getNodes
      .iterator()
      .asScala
      .toList
      .head
      .getConcentration(new SimpleMolecule(Molecules.globalModel))
    val dataset = flUtils.get_dataset(experiment, false)
    val results = flUtils.evaluate(model, dataset, batchSize, experiment)
    val testAccuracy = py"$results[1]"
  }
}
