package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Node, Position, TimeDistribution}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import scala.jdk.CollectionConverters.IteratorHasAsScala
import it.unibo.scafi.Molecules
import scala.util.Random

class KillRandomLeader[T, P <: Position[P]](
  environment: Environment[T, P],
  timeDistribution: TimeDistribution[T],
  seed: Int,
  resilience: Boolean
) extends AbstractGlobalReaction(environment, timeDistribution){

  private val random = new Random(seed)
  private var executed = false

  override protected def executeBeforeUpdateDistribution(): Unit = {
      if (resilience && !executed) {
        executed = true
        val leaders = findLeaders(environment)
        val toBeKilled = random.shuffle(leaders).head
        println(s"[DEBUG] killing node ${toBeKilled.getId}")
        environment.getSimulation.schedule(() => environment.removeNode(toBeKilled))
      }
  }

  private def findLeaders(environment: Environment[T, P]): List[Node[T]] =
    environment
      .getNodes
      .iterator()
      .asScala
      .toList
      .filter(_.getConcentration(new SimpleMolecule(Molecules.isAggregator)).asInstanceOf[Boolean])

}
