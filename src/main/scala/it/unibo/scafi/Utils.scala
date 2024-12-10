package it.unibo.scafi

object Utils {

   def slices(stop: Int, k: Int): List[(Int, Int)] = {
     val start = 0
     val step = (stop - start).toDouble / k
     (0 until k).map { i =>
       val l = math.round(start + i * step).toInt
       val u = math.round(start + (i + 1) * step).toInt
       (l, u)
     }.toList
  }

}
