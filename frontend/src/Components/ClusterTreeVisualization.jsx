"use client"

import { useState, useEffect } from "react"
import { ChevronRight, BarChart2, Layers, ArrowLeft, Eye } from "lucide-react"
import { motion, AnimatePresence } from "framer-motion"

const ClusterTreeVisualization = ({ clusterTree, activeKPI, onClusterSelect }) => {
  const [selectedCluster, setSelectedCluster] = useState(null)
  const [clusterPath, setClusterPath] = useState([])
  const [currentLevel, setCurrentLevel] = useState(0)
  const [currentClusters, setCurrentClusters] = useState([])
  const [clusterDetails, setClusterDetails] = useState(null)
  const [showDetails, setShowDetails] = useState(false)

  useEffect(() => {
    if (clusterTree && activeKPI && clusterTree[activeKPI]) {
      // Start with the root cluster
      const rootCluster = clusterTree[activeKPI]

      // If we're at level 0, show the root cluster itself
      if (currentLevel === 0) {
        setCurrentClusters([rootCluster])
      } else if (clusterPath.length > 0) {
        // If we have a selected path, navigate to that cluster's children
        let currentNode = rootCluster
        for (const pathIndex of clusterPath) {
          if (currentNode.children && currentNode.children[pathIndex]) {
            currentNode = currentNode.children[pathIndex]
          }
        }
        setCurrentClusters(currentNode.children || [])
      } else {
        // Otherwise show the root's children
        setCurrentClusters(rootCluster.children || [])
      }
    }
  }, [clusterTree, activeKPI, clusterPath, currentLevel])

  const navigateToChildren = (cluster, index) => {
    if (cluster.children && cluster.children.length > 0) {
      // Navigate to children
      const newPath = [...clusterPath, index]
      setClusterPath(newPath)
      setCurrentLevel(currentLevel + 1)
      setCurrentClusters(cluster.children)
      if (onClusterSelect) {
        onClusterSelect(cluster, newPath, currentLevel + 1)
      }
    } else {
      // No children to navigate to
      console.log("This cluster has no children")
    }
  }

  const viewClusterDetails = (cluster) => {
    setClusterDetails(cluster)
    setShowDetails(true)
  }

  const handleBackClick = () => {
    if (currentLevel > 0) {
      const newPath = clusterPath.slice(0, -1)
      setClusterPath(newPath)
      setCurrentLevel(currentLevel - 1)

      // Navigate back to the parent cluster's children or root
      if (currentLevel === 1) {
        // Going back to root level
        setCurrentClusters([clusterTree[activeKPI]])
      } else {
        // Going back to a parent's children
        let currentNode = clusterTree[activeKPI]
        for (const pathIndex of newPath) {
          if (currentNode.children && currentNode.children[pathIndex]) {
            currentNode = currentNode.children[pathIndex]
          }
        }
        setCurrentClusters(currentNode.children || [])
      }

      if (onClusterSelect) {
        onClusterSelect(currentNode, newPath, currentLevel - 1)
      }
    }
  }

  const closeDetails = () => {
    setShowDetails(false)
    setClusterDetails(null)
  }

  // Helper function to get a color based on the cluster's contribution
  const getContributionColor = (cluster) => {
    if (!cluster.analysis || !Object.keys(cluster.analysis).length) return "bg-gray-100"

    const firstFeature = Object.keys(cluster.analysis)[0]
    if (!cluster.analysis[firstFeature]?.contributions?.mean_contribution_percentage) return "bg-gray-100"

    const contribution = cluster.analysis[firstFeature].contributions.mean_contribution_percentage

    if (contribution > 0) {
      return contribution > 1000000000000 ? "bg-green-100 border-green-300" : "bg-green-50 border-green-200"
    } else {
      return contribution < -1000000000000 ? "bg-red-100 border-red-300" : "bg-red-50 border-red-200"
    }
  }

  // Helper function to format large numbers
  const formatNumber = (num) => {
    if (Math.abs(num) > 1e12) return (num / 1e12).toFixed(2) + "T"
    if (Math.abs(num) > 1e9) return (num / 1e9).toFixed(2) + "B"
    if (Math.abs(num) > 1e6) return (num / 1e6).toFixed(2) + "M"
    if (Math.abs(num) > 1e3) return (num / 1e3).toFixed(2) + "K"
    return num.toFixed(2)
  }

  // Helper function to get cluster title based on level and index
  const getClusterTitle = (index, level) => {
    if (level === 0) return "Root Cluster"
    return `Cluster ${index + 1}`
  }

  if (!clusterTree || !activeKPI || !clusterTree[activeKPI]) {
    return (
      <div className="flex items-center justify-center h-64 bg-white rounded-lg shadow">
        <p className="text-gray-500">No cluster data available</p>
      </div>
    )
  }

  return (
    <div className="bg-white rounded-lg shadow-md overflow-hidden">
      {/* Header with breadcrumb navigation */}
      <div className="bg-gray-50 p-4 border-b flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Layers className="h-5 w-5 text-purple-500" />
          <h3 className="font-medium">
            Cluster Tree for {activeKPI} (Level {currentLevel})
          </h3>
        </div>

        <div className="flex items-center space-x-2">
          {clusterPath.length > 0 && (
            <button
              onClick={handleBackClick}
              className="flex items-center text-sm text-purple-600 hover:text-purple-800 transition-colors"
            >
              <ArrowLeft className="h-4 w-4 mr-1" />
              Back to Level {currentLevel - 1}
            </button>
          )}
        </div>
      </div>

      {/* Breadcrumb path */}
      <div className="px-4 py-2 bg-gray-50 border-b text-sm flex items-center overflow-x-auto">
        <span
          className="text-purple-600 hover:text-purple-800 cursor-pointer"
          onClick={() => {
            setClusterPath([])
            setCurrentLevel(0)
            setCurrentClusters([clusterTree[activeKPI]])
          }}
        >
          Root
        </span>

        {clusterPath.map((pathIndex, idx) => (
          <div key={idx} className="flex items-center">
            <ChevronRight className="h-4 w-4 mx-1 text-gray-400" />
            <span
              className="text-purple-600 hover:text-purple-800 cursor-pointer"
              onClick={() => {
                const newPath = clusterPath.slice(0, idx + 1)
                setClusterPath(newPath)
                setCurrentLevel(idx + 1)

                // Navigate to this specific level
                let currentNode = clusterTree[activeKPI]
                for (const pIdx of newPath) {
                  if (currentNode.children && currentNode.children[pIdx]) {
                    currentNode = currentNode.children[pIdx]
                  }
                }
                setCurrentClusters(currentNode.children || [])
              }}
            >
              Cluster {pathIndex + 1}
            </span>
          </div>
        ))}
      </div>

      {/* Clusters grid */}
      <div className="p-4">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {currentClusters.map((cluster, index) => (
            <motion.div
              key={cluster.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.3, delay: index * 0.05 }}
              className={`border rounded-lg p-4 hover:shadow-md transition-shadow ${getContributionColor(cluster)}`}
            >
              <div className="flex justify-between items-start mb-3">
                <div className="flex items-center">
                  <BarChart2 className="h-5 w-5 text-purple-500 mr-2" />
                  <h4 className="font-medium">{getClusterTitle(index, currentLevel)}</h4>
                </div>
                {cluster.children && cluster.children.length > 0 ? (
                  <div className="flex items-center text-xs text-gray-500 bg-white bg-opacity-60 px-2 py-1 rounded-full">
                    <Layers className="h-3 w-3 mr-1" />
                    {cluster.children.length} subclusters
                  </div>
                ) : (
                  <div className="text-xs text-gray-500 bg-white bg-opacity-60 px-2 py-1 rounded-full">Leaf node</div>
                )}
              </div>

              <div className="space-y-2">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Size:</span>
                  <span className="font-medium">{cluster.size} records</span>
                </div>

                {cluster.score !== null && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Score:</span>
                    <span className="font-medium">{cluster.score?.toFixed(4)}</span>
                  </div>
                )}

                {/* Show the first feature's contribution if available */}
                {cluster.analysis && Object.keys(cluster.analysis).length > 0 && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <div className="text-xs text-gray-500 mb-1">Top Feature:</div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm truncate max-w-[70%]">{Object.keys(cluster.analysis)[0]}</span>
                      <span className="text-sm font-medium">
                        {cluster.analysis[Object.keys(cluster.analysis)[0]]?.segment?.mean?.toFixed(4) || "N/A"}
                      </span>
                    </div>
                  </div>
                )}
              </div>

              <div className="mt-3 flex justify-between">
                <button
                  onClick={() => viewClusterDetails(cluster)}
                  className="text-purple-600 text-sm flex items-center hover:text-purple-800"
                >
                  <Eye className="h-4 w-4 mr-1" /> View Details
                </button>

                {cluster.children && cluster.children.length > 0 && (
                  <button
                    onClick={() => navigateToChildren(cluster, index)}
                    className="text-purple-600 text-sm flex items-center hover:text-purple-800"
                  >
                    Explore <ChevronRight className="h-4 w-4 ml-1" />
                  </button>
                )}
              </div>
            </motion.div>
          ))}
        </div>

        {currentClusters.length === 0 && (
          <div className="flex items-center justify-center h-40">
            <p className="text-gray-500">No clusters available at this level</p>
          </div>
        )}
      </div>

      {/* Cluster details modal */}
      <AnimatePresence>
        {showDetails && clusterDetails && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
            onClick={closeDetails}
          >
            <motion.div
              initial={{ scale: 0.9, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.9, y: 20 }}
              className="bg-white rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="p-6">
                <div className="flex justify-between items-center mb-6">
                  <h3 className="text-xl font-semibold text-gray-800">
                    {currentLevel === 0 ? "Root Cluster Details" : `Cluster Details (Level ${clusterDetails.level})`}
                  </h3>
                  <button onClick={closeDetails} className="text-gray-400 hover:text-gray-600">
                    <svg
                      xmlns="http://www.w3.org/2000/svg"
                      className="h-6 w-6"
                      fill="none"
                      viewBox="0 0 24 24"
                      stroke="currentColor"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-700 mb-3">Cluster Information</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600">ID:</span>
                        <span className="font-mono text-sm">{clusterDetails.id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Size:</span>
                        <span>{clusterDetails.size} records</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Level:</span>
                        <span>{clusterDetails.level}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Path:</span>
                        <span className="font-mono text-sm">{clusterDetails.path.join(" → ")}</span>
                      </div>
                      {clusterDetails.score !== null && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Score:</span>
                          <span>{clusterDetails.score?.toFixed(4)}</span>
                        </div>
                      )}
                      {clusterDetails.children && (
                        <div className="flex justify-between">
                          <span className="text-gray-600">Subclusters:</span>
                          <span>{clusterDetails.children.length}</span>
                        </div>
                      )}
                    </div>
                  </div>

                  <div className="bg-gray-50 p-4 rounded-lg">
                    <h4 className="font-medium text-gray-700 mb-3">KPI Information</h4>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600">KPI Column:</span>
                        <span>{clusterDetails.kpi_column}</span>
                      </div>
                      {clusterDetails.analysis && Object.keys(clusterDetails.analysis).length > 0 && (
                        <div>
                          <div className="text-gray-600 mb-2 mt-4">Feature Analysis:</div>
                          <div className="max-h-40 overflow-y-auto pr-2">
                            {Object.entries(clusterDetails.analysis).map(([feature, data]) => (
                              <div key={feature} className="mb-3 pb-3 border-b border-gray-200 last:border-0">
                                <div className="font-medium text-sm mb-1">{feature}</div>
                                <div className="grid grid-cols-2 gap-2 text-sm">
                                  <div>
                                    <span className="text-gray-500">Mean:</span>{" "}
                                    <span className="font-mono">{data.segment?.mean?.toFixed(6) || "N/A"}</span>
                                  </div>
                                  <div>
                                    <span className="text-gray-500">Sum:</span>{" "}
                                    <span className="font-mono">{formatNumber(data.segment?.sum || 0)}</span>
                                  </div>
                                  {data.contributions && (
                                    <div className="col-span-2">
                                      <span className="text-gray-500">Contribution:</span>{" "}
                                      <span className="font-mono">
                                        {formatNumber(data.contributions.mean_contribution_percentage)}%
                                      </span>
                                    </div>
                                  )}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                <div className="bg-gray-50 p-4 rounded-lg">
                  <h4 className="font-medium text-gray-700 mb-3">Data Points</h4>
                  <div className="text-sm text-gray-600 mb-2">
                    This cluster contains {clusterDetails.indices.length} data points.
                  </div>
                  <div className="max-h-40 overflow-y-auto">
                    <div className="grid grid-cols-4 sm:grid-cols-6 md:grid-cols-8 gap-2">
                      {clusterDetails.indices.slice(0, 100).map((index) => (
                        <div key={index} className="bg-white p-2 rounded border text-sm text-center">
                          {index}
                        </div>
                      ))}
                      {clusterDetails.indices.length > 100 && (
                        <div className="col-span-full text-center text-gray-500 text-sm py-2">
                          + {clusterDetails.indices.length - 100} more indices
                        </div>
                      )}
                    </div>
                  </div>
                </div>

                {clusterDetails.children && clusterDetails.children.length > 0 && (
                  <div className="mt-6 flex justify-end">
                    <button
                      onClick={() => {
                        // Find the index of this cluster in the current level
                        const index = currentClusters.findIndex((c) => c.id === clusterDetails.id)
                        if (index !== -1) {
                          closeDetails()
                          navigateToChildren(clusterDetails, index)
                        }
                      }}
                      className="px-4 py-2 bg-purple-500 text-white rounded-md hover:bg-purple-600 transition-colors"
                    >
                      Explore Subclusters ({clusterDetails.children.length})
                    </button>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export default ClusterTreeVisualization
