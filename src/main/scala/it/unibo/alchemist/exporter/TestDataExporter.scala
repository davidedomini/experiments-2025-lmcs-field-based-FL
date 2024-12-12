package it.unibo.alchemist.exporter

import com.github.tototoshi.csv.CSVWriter
import java.io.File

object TestDataExporter {

  def CSVExport(data: List[Double], path: String): Unit = {
    val file = new File(s"$path.csv")
    val writer = CSVWriter.open(file)
    val header = data.zipWithIndex.map { case (_, i) => s"Node-$i" }
    writer.writeAll(List(header, data))
    writer.close()
  }

}
