package ml

object common {
  private type OptionMap = Map[Symbol, Any]

  val usage = """
    Usage: function [--ipDataPath str] [--ipModelPath str] [--opDataPath str] [--opModelPath str]
  """

  def args_parse(args: Array[String]): OptionMap = {
    if (args.length == 0) println(usage)
    val arglist = args.toList

    def nextOption(map : OptionMap, list: List[String]) : OptionMap = {
      def isSwitch(s : String) = (s(0) == '-')
      list match {
        case Nil => map

        case "--ipDataPath" :: value :: tail =>
          nextOption(map ++ Map('ipDataPath -> value.toString), tail)

        case "--ipModelPath" :: value :: tail =>
          nextOption(map ++ Map('ipModelPath -> value.toString), tail)

        case "--opDataPath" :: value :: tail =>
          nextOption(map ++ Map('opDataPath -> value.toString), tail)

        case "--opModelPath" :: value :: tail =>
          nextOption(map ++ Map('opModelPath -> value.toString), tail)

        case option :: tail => println("Unknown option "+option)
          sys.exit(1)
      }
    }
    val options = nextOption(Map(), arglist)
    println(options)

    return options
  }

  val feature_var = Array("locale", "day_of_week", "hour_of_day", "agent_id", "entry_page", "traffic_type",
    "session_duration", "countLength", "logLikelihood", "durPerPage", "logPosterior", "weekend", "am",
    "peakHours", "evening", "sleepHours", "durCo", "cocounts", "avgCocounts")
}
