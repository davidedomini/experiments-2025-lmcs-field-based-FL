package it.unibo.learning.model

import me.shadaj.scalapy.py

case class Dataset(
  areaId: Int,
  trainingData: py.Dynamic,
  validationData: py.Dynamic
)
