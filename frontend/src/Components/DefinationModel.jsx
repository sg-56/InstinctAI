"use client"
import axios from "axios"
import { useEffect, useState } from "react"
import { useParams } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import { X, Info, Loader2 } from "lucide-react"

const DefinationModel = ({ setIsOpen, kpi, clusterNo, path }) => {
  const baseUrl = import.meta.env.VITE_BASE_URL
  const [data, setData] = useState([])
  const [loading, setLoading] = useState(true)
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

  const checkSubclusterStatus = (task_id2) => {
    return new Promise((resolve, reject) => {
      const intervalId = setInterval(async () => {
        try {
          const subclusterStatus = await axios.get(`${baseUrl}/projects/tasks/${task_id2}/status`)
          if (subclusterStatus.data.status === "SUCCESS") {
            clearInterval(intervalId)
            resolve()
          } else if (subclusterStatus.data.status === "FAILURE") {
            reject(new Error("Subcluster task failed"))
            clearInterval(intervalId)
            throw new Error("Processing subcluster failed")
          }
        } catch (err) {
          clearInterval(intervalId)
          reject(err)
        }
      }, 1000)
    })
  }

  const fetchData = async () => {
    try {
      setLoading(true)
      const response = await axios.post(
        `${baseUrl}/projects/${project_id}/clusters/defination`,
        {
          path: [],
          kpi: kpi,
          cluster_no: clusterNo,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        },
      )
      response.data.sort((a, b) => b.abs_z_score - a.abs_z_score)
      response.data = response.data.slice(0, 5)
      setData(response.data)
    } catch (error) {
      console.error("Error fetching data:", error)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchData()
  }, [])

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
        variants={backdropVariants}
        initial="hidden"
        animate="visible"
        exit="hidden"
        onClick={() => setIsOpen(false)}
      >
        <motion.div
          className="bg-white dark:bg-gray-800 rounded-xl shadow-xl w-full max-w-md mx-4 overflow-hidden"
          variants={modalVariants}
          initial="hidden"
          animate="visible"
          exit="hidden"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Modal Header */}
          <div className="flex justify-between items-center p-4 border-b border-gray-200 dark:border-gray-700">
            <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2">
              <Info className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
              Definition
            </h2>
            <motion.button
              onClick={() => setIsOpen(false)}
              className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-full p-1"
              whileHover={{ scale: 1.1 }}
              whileTap={{ scale: 0.9 }}
            >
              <X className="h-5 w-5" />
            </motion.button>
          </div>

          {/* Table */}
          <div className="p-4 max-h-[80vh] overflow-y-auto">
            {loading ? (
              <div className="flex flex-col items-center justify-center py-8">
                <Loader2 className="h-8 w-8 text-indigo-500 dark:text-indigo-400 animate-spin mb-4" />
                <p className="text-gray-600 dark:text-gray-400">Loading definition data...</p>
              </div>
            ) : (
              <table className="w-full border-collapse">
                <thead className="sticky top-0 bg-white dark:bg-gray-800">
                  <tr>
                    <th className="text-left py-3 px-4 font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700">
                      Parameter
                    </th>
                    <th className="text-right py-3 px-4 font-medium text-gray-700 dark:text-gray-300 border-b border-gray-200 dark:border-gray-700">
                      Weight
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {data?.map((row, index) => (
                    <motion.tr
                      key={index}
                      className="border-b border-gray-100 dark:border-gray-700 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-700/30"
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      transition={{ delay: index * 0.05 }}
                    >
                      <td className="py-3 px-4 text-gray-800 dark:text-gray-300">{row.feature}</td>
                      <td className="py-3 px-4 text-gray-800 dark:text-gray-300 text-right font-medium">
                        {row.abs_z_score.toFixed(3)}
                      </td>
                    </motion.tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>

          {/* Footer */}
          <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              These weights indicate the relative importance of each parameter in defining this segment.
            </p>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

export default DefinationModel

