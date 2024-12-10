package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.layers.PhenomenaDistribution
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.scafi.Molecules
import scala.jdk.CollectionConverters.IteratorHasAsScala

class PhenomenaToDataset[T, P <: Position[P]](
  environment: Environment[T, P],
  timeDistribution: TimeDistribution[T]
) extends AbstractGlobalReaction(environment, timeDistribution) {

  private lazy val phenomenaDistributionLayer = environment
    .getLayers
    .iterator()
    .asScala
    .toList
    .filter(_.isInstanceOf[PhenomenaDistribution[P]])
    .map(_.asInstanceOf[PhenomenaDistribution[P]])
    .head

  override protected def executeBeforeUpdateDistribution(): Unit = {
    val nodeToId = nodes.map { node =>
      val position = environment.getPosition(node)
      val areaId = phenomenaDistributionLayer.getAreaIdByPosition(position)
      (node, areaId)
    }
    val idNodesMapping = nodeToId.groupMap(_._2)(_._1)
    nodes.foreach { node =>
      val areaId = idNodesMapping
        .find {case (_, nodes) => nodes.contains(node) }
        .getOrElse(throw new IllegalStateException(s"Area id for node $node not found"))
        ._1
      val neighs = idNodesMapping.getOrElse(areaId, throw new IllegalStateException(s"Nodes not found for area id $areaId")).size
      val idInArea = idNodesMapping
        .getOrElse(areaId, throw new IllegalStateException(s"Nodes not found for area id $areaId"))
        .indexOf(node)
      node.setConcentration(new SimpleMolecule(Molecules.neighbors), neighs.asInstanceOf[T])
      node.setConcentration(new SimpleMolecule(Molecules.idInArea), idInArea.asInstanceOf[T])
    }

  }

}