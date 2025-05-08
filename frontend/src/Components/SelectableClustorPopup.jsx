"use client"

import axios from "axios"
import { useState, useEffect } from "react"
import { useParams } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import { X, Loader2, Plus, Info, BarChart4 } from "lucide-react"

export default function SelectableClusterPopup({
  setIsOpen1,
  selectedCluster,
  setSelectedCluster,
  kpi,
  feature,
  path,
}) {
  const [clusters, setClusters] = useState([])
  const [currentPage, setCurrentPage] = useState(1)
  const [isLoading, setIsLoading] = useState(false)
  const clustersPerPage = 9
  const baseUrl = import.meta.env.VITE_BASE_URL
  let clustorLength = 1
  const { project_id } = useParams()

  // Animation variants
  const backdropVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
  }

  const modalVariants = {
    hidden: { opacity: 0, y: 50, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1, transition: { type: "spring", damping: 25, stiffness: 300 } },
  }

  const cardVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: (i) => ({
      opacity: 1,
      y: 0,
      transition: {
        delay: i * 0.05,
        duration: 0.3,
      },
    }),
  }

  useEffect(() => {
    const fetchData = async () => {
      setIsLoading(true)
      try {
        const result = await axios.post(`${baseUrl}/projects/${project_id}/features/label`, {
          path: [0],
          kpi: feature,
        })

        const { task_id } = result.data

        const checkStatus = async () => {
          try {
            const statusResponse = await axios.get(`${baseUrl}/projects/tasks/${task_id}/status`, {
              headers: {
                "Content-Type": "application/json",
              },
            })
            return statusResponse.data
          } catch (error) {
            console.error("Error checking task status:", error)
            throw error
          }
        }

        const pollStatus = async () => {
          try {
            const status = await checkStatus()
            if (status.status === "SUCCESS") {
              const response = await axios.post(`${baseUrl}/projects/${project_id}/features/weight/label`, {
                params: {
                  path: [0],
                  kpi: feature,
                },
                headers: {
                  "Content-Type": "application/json",
                },
              })

              let resultData = response.data
              resultData.sort((a, b) => b.Impact_Score - a.Impact_Score)
              resultData = resultData.slice(0, 5)
              setClusters([{ id: 1, parameters: resultData }])
              clustorLength = resultData.length
            } else {
              throw new Error(`Task failed with status: ${status.status}`)
            }
          } catch (error) {
            console.error("Error in polling:", error)
            throw error
          }
        }

        // Start polling
        await pollStatus()
      } catch (error) {
        console.error("Error fetching data:", error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [project_id, path, feature])

  const handleLoadMore = () => {
    if (currentPage * clustersPerPage < clustorLength) {
      setCurrentPage((prevPage) => prevPage + 1)
    }
  }

  const handleSelectCluster = (clusterId) => {
    setSelectedCluster(clusterId)
  }

  const displayedClusters = clusters.slice(0, currentPage * clustersPerPage)

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
        variants={backdropVariants}
        initial="hidden"
        animate="visible"
        exit="hidden"
        onClick={() => setIsOpen1(false)}
      >
        <motion.div
          className="relative bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-4xl mx-4 p-6 overflow-hidden"
          variants={modalVariants}
          initial="hidden"
          animate="visible"
          exit="hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Header */}
          <div className="flex justify-between items-center mb-6">
            <h2 className="text-xl font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <BarChart4 className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
              Select Cluster for {feature}
            </h2>
            <motion.button
              onClick={() => setIsOpen1(false)}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-full p-1"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <X className="h-5 w-5" />
            </motion.button>
          </div>

          {/* Content */}
          <div className="max-h-[70vh] overflow-y-auto pr-2 custom-scrollbar">
            {isLoading ? (
              <div className="flex flex-col items-center justify-center py-12">
                <Loader2 className="h-10 w-10 text-indigo-500 dark:text-indigo-400 animate-spin mb-4" />
                <p className="text-gray-600 dark:text-gray-400">Loading clusters...</p>
              </div>
            ) : (
              <>
                <div
                  className={`grid gap-6 ${displayedClusters.length === 1 ? "grid-cols-1 place-items-center" : "grid-cols-1 sm:grid-cols-2 lg:grid-cols-3"}`}
                >
                  {displayedClusters.map((cluster, index) => (
                    <motion.div
                      key={index}
                      custom={index}
                      variants={cardVariants}
                      initial="hidden"
                      animate="visible"
                      className={`cursor-pointer rounded-lg border w-full max-w-[400px] transition-all hover:shadow-md
                        ${
                          selectedCluster === cluster.id
                            ? "border-indigo-500 dark:border-indigo-400 ring-2 ring-indigo-500 dark:ring-indigo-400"
                            : "border-gray-200 dark:border-gray-700"
                        }`}
                      onClick={() => handleSelectCluster(cluster.id)}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      <div className="bg-indigo-600 dark:bg-indigo-700 text-white rounded-t-lg py-3 px-4">
                        <h3 className="text-sm font-medium flex items-center justify-between">
                          <span>Cluster {cluster.id}</span>
                          {selectedCluster === cluster.id && (
                            <span className="text-xs bg-white text-indigo-600 px-2 py-0.5 rounded-full">Selected</span>
                          )}
                        </h3>
                      </div>

                      {/* Parameters Table */}
                      <div className="p-4">
                        <table className="w-full">
                          <thead>
                            <tr>
                              <th className="py-2 text-left text-xs font-medium text-gray-500 dark:text-gray-400">
                                Parameters
                              </th>
                              <th className="py-2 text-right text-xs font-medium text-gray-500 dark:text-gray-400">
                                Weight
                              </th>
                            </tr>
                          </thead>
                          <tbody>
                            {cluster.parameters?.map((param, paramIndex) => (
                              <motion.tr
                                key={paramIndex}
                                className="border-t border-gray-100 dark:border-gray-700"
                                initial={{ opacity: 0 }}
                                animate={{ opacity: 1 }}
                                transition={{ delay: 0.1 + paramIndex * 0.05 }}
                              >
                                <td className="py-2 text-sm text-gray-700 dark:text-gray-300">{param.Feature}</td>
                                <td className="py-2 text-right text-sm font-medium text-gray-700 dark:text-gray-300">
                                  {param.Impact_Score.toFixed(3)}
                                </td>
                              </motion.tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </motion.div>
                  ))}
                </div>

                {/* Load More Button */}
                {currentPage * clustersPerPage < clustorLength && (
                  <div className="mt-6 flex justify-center">
                    <motion.button
                      onClick={handleLoadMore}
                      className="flex items-center justify-center gap-2 px-4 py-2 rounded-full bg-indigo-50 dark:bg-indigo-900/30 text-indigo-600 dark:text-indigo-400 font-medium transition-colors hover:bg-indigo-100 dark:hover:bg-indigo-900/50"
                      whileHover={{ scale: 1.05 }}
                      whileTap={{ scale: 0.95 }}
                    >
                      <Plus className="h-4 w-4" />
                      <span>Load More</span>
                    </motion.button>
                  </div>
                )}

                {/* Pagination Info */}
                <div className="mt-4 text-center text-sm text-gray-600 dark:text-gray-400">
                  Showing {Math.min(currentPage * clustersPerPage, clustorLength)} of {clustorLength} clusters
                </div>
              </>
            )}
          </div>

          {/* Footer */}
          <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700 flex justify-between items-center">
            <div className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
              <Info className="h-4 w-4" />
              <span>Select a cluster to view detailed parameters</span>
            </div>
            <motion.button
              onClick={() => setIsOpen1(false)}
              className="px-4 py-2 bg-indigo-600 dark:bg-indigo-700 text-white rounded-md font-medium"
              whileHover={{ scale: 1.05 }}
              whileTap={{ scale: 0.95 }}
            >
              Confirm Selection
            </motion.button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

