package it.unibo.alchemist.model.implementations.reactions

import it.unibo.alchemist.model.{Environment, Position, TimeDistribution}
import it.unibo.alchemist.model.molecules.SimpleMolecule
import it.unibo.interop.PythonModules.flUtils
import me.shadaj.scalapy.py.SeqConverters
import scala.language.implicitConversions
import it.unibo.scafi.Molecules
import me.shadaj.scalapy.py

class FedNovaCentralizedServer[T, P <: Position[P]](
 environment: Environment[T, P],
 timeDistribution: TimeDistribution[T]
) extends AbstractGlobalReaction(environment, timeDistribution){

  private implicit def toMolecule(name: String): SimpleMolecule = new SimpleMolecule(name)

  override protected def executeBeforeUpdateDistribution(): Unit = {

    var clientState = List.empty[py.Dynamic]
    var clientNData = List.empty[Int]
    var clientCoeff = List.empty[Double]
    var clientNormGrad = List.empty[py.Dynamic]

    val oldGlobalModel = nodes.head.getConcentration(Molecules.globalModel).asInstanceOf[py.Dynamic]

    nodes.foreach { node =>
      val localModel = node.getConcentration(Molecules.localModel).asInstanceOf[py.Dynamic]
      val nData = node.getConcentration(Molecules.nData).asInstanceOf[Int]
      val tau = node.getConcentration(Molecules.tau).asInstanceOf[Double]
      val normGrad = node.getConcentration(Molecules.normGrad).asInstanceOf[py.Dynamic]

      clientState = clientState.appended(localModel)
      clientNData = clientNData.appended(nData)
      clientCoeff = clientCoeff.appended(tau)
      clientNormGrad = clientNormGrad.appended(normGrad)
    }

    val globalModel = flUtils
      .aggregate_fed_nova(oldGlobalModel, clientState.toPythonProxy, clientNData.toPythonProxy, clientCoeff.toPythonProxy, clientNormGrad.toPythonProxy, nodes.size)

    nodes.foreach { node =>
      node.setConcentration(Molecules.globalModel, globalModel.asInstanceOf[T])
    }

  }

}
