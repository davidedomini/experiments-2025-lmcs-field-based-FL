package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.interop.PythonModules._
import it.unibo.scafi.Molecules
import me.shadaj.scalapy.py.SeqConverters


class ModelAndSeedInitialization[T, P <: Position[P]](
 environment: Environment[T, P],
 timeDistribution: TimeDistribution[T],
 seed: Int,
 experiment: String
) extends AbstractGlobalReaction(environment, timeDistribution)  {

  override protected def executeBeforeUpdateDistribution(): Unit = {
    flUtils.seed_everything(seed)
    val model = flUtils
      .instantiate_model(List.empty[Int].toPythonProxy, experiment, false)
      .state_dict()
    nodes.foreach { node =>
      node.setConcentration(new SimpleMolecule(Molecules.globalModel), model.asInstanceOf[T])
    }
  }
}
