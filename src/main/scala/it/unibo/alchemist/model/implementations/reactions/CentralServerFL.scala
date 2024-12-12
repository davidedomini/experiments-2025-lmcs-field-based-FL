package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.interop.PythonModules._
import it.unibo.scafi.Molecules
import me.shadaj.scalapy.py
import me.shadaj.scalapy.py.SeqConverters

class CentralServerFL [T, P <: Position[P]](
 environment: Environment[T, P],
 timeDistribution: TimeDistribution[T]
) extends AbstractGlobalReaction(environment, timeDistribution){

  override protected def executeBeforeUpdateDistribution(): Unit = {

    val models = nodes
      .map { node => node.getConcentration(new SimpleMolecule(Molecules.localModel)) }
      .map { model => model.asInstanceOf[py.Dynamic]}

    val globalModel = flUtils.average_weights(models.toPythonProxy, List.fill(models.length)(1.0))

    nodes.foreach { node =>
      node.setConcentration(new SimpleMolecule(Molecules.globalModel), globalModel.asInstanceOf[T])
    }

  }

}

