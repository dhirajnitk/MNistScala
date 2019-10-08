name := "Scala"

version := "1.0"

scalaVersion := "2.12.2"

classpathTypes += "maven-plugin"
excludeFilter in unmanagedSources := "OMP4JUtility.java"

libraryDependencies += "org.nd4j" % "nd4j-native" % "0.4-rc3.10" classifier "" classifier "windows-x86_64"
