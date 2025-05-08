"use client"

import { useState } from "react"
import { useNavigate } from "react-router-dom"
import axios from "axios"
import { motion, AnimatePresence } from "framer-motion"
import { X, Check, Database, Loader2 } from "lucide-react"

const WorkbenchModal = ({
  categorical_columns,
  currentLevel,
  currentPath,
  activeKPI,
  showModal,
  setShowModal,
  project_id,
}) => {
  const baseUrl = import.meta.env.VITE_BASE_URL
  const navigate = useNavigate()
  const [selectedColumns, setSelectedColumns] = useState([])
  const [loading, setLoading] = useState(false)

  // Animation variants
  const backdropVariants = {
    hidden: { opacity: 0 },
    visible: { opacity: 1 },
  }

  const modalVariants = {
    hidden: { opacity: 0, y: 50, scale: 0.95 },
    visible: { opacity: 1, y: 0, scale: 1, transition: { type: "spring", damping: 25, stiffness: 300 } },
  }

  const handleChecks = (e) => {
    const { checked, value } = e.target
    setSelectedColumns((prev) => (checked ? [...prev, value] : prev.filter((col) => col !== value)))
  }

  const handleSubmit = async () => {
    try {
      setLoading(true)
      const encodingPromises = selectedColumns.map((column) =>
        axios.post(`${baseUrl}/projects/${project_id}/time-series/encoded-columns`, {
          project_id,
          column_name: column,
          level: currentLevel,
          path: currentPath,
        }),
      )
      const responses = await Promise.all(encodingPromises)
      const newEncodedCols = {}

      responses.forEach((response, index) => {
        const column = selectedColumns[index]
        newEncodedCols[column] = response.data.categorical_column
      })

      setShowModal(false)
      navigate(`/projects/${project_id}/workbench`, {
        state: {
          currentLevel,
          currentPath,
          activeKPI,
          encodedCols: newEncodedCols,
          categorical_columns,
        },
      })
    } catch (error) {
      console.error("Error submitting data:", error.response?.data?.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <AnimatePresence>
      <motion.div
        className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
        variants={backdropVariants}
        initial="hidden"
        animate="visible"
        exit="hidden"
        onClick={() => !loading && setShowModal(false)}
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
              <Database className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
              Select Categorical Columns
            </h2>
            {!loading && (
              <motion.button
                onClick={() => setShowModal(false)}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-full p-1"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                <X className="h-5 w-5" />
              </motion.button>
            )}
          </div>

          {/* Modal Content */}
          <div className="p-4">
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              Select the categorical columns you want to include in your workbench analysis:
            </p>

            <div className="max-h-[40vh] overflow-y-auto pr-2 custom-scrollbar">
              <div className="space-y-3">
                {categorical_columns.map((column, index) => (
                  <motion.div
                    key={index}
                    className="flex items-center gap-3 p-2 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700/30"
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.03 }}
                    whileHover={{ x: 4 }}
                  >
                    <div className="relative">
                      <input
                        type="checkbox"
                        id={column}
                        name={column}
                        value={column}
                        onChange={handleChecks}
                        className="h-4 w-4 rounded border-gray-300 text-indigo-600 focus:ring-indigo-500 dark:border-gray-600 dark:bg-gray-700 dark:focus:ring-indigo-400"
                        disabled={loading}
                      />
                      <div
                        className={`absolute inset-0 rounded flex items-center justify-center transition-opacity ${
                          selectedColumns.includes(column) ? "opacity-100" : "opacity-0"
                        }`}
                      >
                        <Check className="h-3 w-3 text-indigo-600 dark:text-indigo-400" />
                      </div>
                    </div>
                    <label htmlFor={column} className="text-sm text-gray-700 dark:text-gray-300 cursor-pointer flex-1">
                      {column}
                    </label>
                  </motion.div>
                ))}
              </div>
            </div>
          </div>

          {/* Modal Footer */}
          <div className="flex justify-end gap-3 p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
            {!loading && (
              <motion.button
                type="button"
                onClick={() => setShowModal(false)}
                className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
                disabled={loading}
              >
                Cancel
              </motion.button>
            )}
            <motion.button
              onClick={handleSubmit}
              className={`px-4 py-2 text-sm font-medium text-white rounded-md transition-colors flex items-center gap-2 ${
                loading
                  ? "bg-indigo-400 dark:bg-indigo-600 cursor-not-allowed"
                  : "bg-indigo-600 dark:bg-indigo-700 hover:bg-indigo-700 dark:hover:bg-indigo-800"
              }`}
              whileHover={loading ? {} : { scale: 1.05 }}
              whileTap={loading ? {} : { scale: 0.95 }}
              disabled={loading}
            >
              {loading ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  <span>Processing...</span>
                </>
              ) : (
                <span>Submit</span>
              )}
            </motion.button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  )
}

export default WorkbenchModal

