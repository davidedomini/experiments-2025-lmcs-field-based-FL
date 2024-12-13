package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.scafi.Molecules

class CheckTheLeader [T, P <: Position[P]](
  environment: Environment[T, P],
  timeDistribution: TimeDistribution[T]
) extends AbstractGlobalReaction(environment, timeDistribution) {

  override protected def executeBeforeUpdateDistribution(): Unit = {

    val leaders = nodes
      .filter { node => node.getConcentration(new SimpleMolecule(Molecules.isAggregator)).asInstanceOf[Boolean] }
      .zipWithIndex

    nodes.foreach { node =>
      val leader = leaders.minBy { case (l, _) => environment.getDistanceBetweenNodes(l, node) }
      val leaderId = (leader._2 * (10/3).toDouble).toInt
      node.setConcentration(new SimpleMolecule(Molecules.federation), leaderId.asInstanceOf[T])
    }

  }

}