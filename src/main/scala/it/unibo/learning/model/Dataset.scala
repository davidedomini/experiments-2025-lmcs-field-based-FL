package it.unibo.learning.model

import me.shadaj.scalapy.py

case class Dataset(
  areaId: Int,
  trainingData: List[Int],
  validationData: List[Int]
)
