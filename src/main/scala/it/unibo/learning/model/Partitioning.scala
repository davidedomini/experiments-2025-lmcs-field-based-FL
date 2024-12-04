package it.unibo.learning.model

sealed trait Partitioning

case object Hard extends Partitioning
case class Dirichlet(beta: Double) extends Partitioning