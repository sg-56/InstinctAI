"use client"
import { Clock, Layers, Target } from "lucide-react"
import { motion } from "framer-motion"
import { useSelector, useDispatch } from "react-redux"

const ClusterHistorySection = ({ selectedCell, currentLevel, handleHistoryClick }) => {
  const dispatch = useDispatch()
  const { clusterHistory, selectedIndex } = useSelector((state) => state.cluster)

  const handleClick = (item, index) => {
    handleHistoryClick(item, index)
  }

  // Animation variants
  const listVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: {
        staggerChildren: 0.05,
      },
    },
  }

  const itemVariants = {
    hidden: { x: -20, opacity: 0 },
    visible: { x: 0, opacity: 1 },
  }

  return (
    <div className="w-full h-full bg-white dark:bg-gray-800 overflow-y-auto">
      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 flex items-center gap-2">
          <Clock className="h-5 w-5 text-indigo-500 dark:text-indigo-400" />
          Segment History
        </h3>
      </div>

      <div className="p-4">
        {selectedCell && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-6 p-4 bg-indigo-50 dark:bg-indigo-900/20 rounded-lg shadow-sm border border-indigo-100 dark:border-indigo-800"
          >
            <h4 className="font-medium text-indigo-800 dark:text-indigo-300 mb-2 flex items-center gap-2">
              <Target className="h-4 w-4" />
              Current Selection:
            </h4>
            <div className="space-y-2 text-sm">
              <p className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Level:</span>
                <span className="font-medium text-gray-900 dark:text-gray-200">{currentLevel + 1}</span>
              </p>
              <p className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Parameter:</span>
                <span className="font-medium text-gray-900 dark:text-gray-200">{selectedCell.feature}</span>
              </p>
              <p className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Value:</span>
                <span className="font-medium text-gray-900 dark:text-gray-200">
                  {typeof selectedCell?.value === "number"
                    ? `${selectedCell?.value?.toFixed(4)}`
                    : `${selectedCell?.value} - ${selectedCell?.percentage}%`}
                </span>
              </p>
              <p className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Segment:</span>
                <span className="font-medium text-gray-900 dark:text-gray-200">{selectedCell.clusterIndex + 1}</span>
              </p>
            </div>
          </motion.div>
        )}

        {clusterHistory.length > 0 ? (
          <motion.div className="space-y-2" variants={listVariants} initial="hidden" animate="visible">
            {clusterHistory.map((item, index) => (
              <motion.div
                key={index}
                variants={itemVariants}
                onClick={() => handleClick(item, index)}
                className={`p-4 rounded-lg cursor-pointer transition-all ${
                  index === selectedIndex
                    ? "bg-indigo-100 dark:bg-indigo-900/30 border-l-4 border-indigo-500 dark:border-indigo-400"
                    : "bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700 shadow-sm"
                }`}
                whileHover={{ scale: 1.02, x: 4 }}
                whileTap={{ scale: 0.98 }}
              >
                <div className="flex items-start gap-3">
                  <div
                    className={`mt-0.5 ${
                      item.level === currentLevel
                        ? "text-indigo-600 dark:text-indigo-400"
                        : "text-gray-400 dark:text-gray-500"
                    }`}
                  >
                    <Layers className="w-5 h-5" />
                  </div>
                  <div className="flex-1">
                    <p
                      className={`font-medium ${
                        item.level === currentLevel
                          ? "text-indigo-800 dark:text-indigo-300"
                          : "text-gray-800 dark:text-gray-300"
                      }`}
                    >
                      Segment {item.cluster}
                    </p>
                    <div className="mt-1 space-y-1">
                      <p
                        className={`text-sm flex justify-between ${
                          item.level === currentLevel
                            ? "text-indigo-600 dark:text-indigo-400"
                            : "text-gray-600 dark:text-gray-400"
                        }`}
                      >
                        <span>Value:</span>
                        <span className="font-medium">
                          {typeof item.value === "number"
                            ? Number(item.value).toFixed(4)
                            : `${item?.value} - ${item?.percentage}%`}
                        </span>
                      </p>
                      <p
                        className={`text-sm flex justify-between ${
                          item.level === currentLevel
                            ? "text-indigo-600 dark:text-indigo-400"
                            : "text-gray-600 dark:text-gray-400"
                        }`}
                      >
                        <span>Level:</span>
                        <span className="font-medium">{item.level + 1}</span>
                      </p>
                      <p
                        className={`text-sm flex justify-between ${
                          item.level === currentLevel
                            ? "text-indigo-600 dark:text-indigo-400"
                            : "text-gray-600 dark:text-gray-400"
                        }`}
                      >
                        <span>Parameter:</span>
                        <span className="font-medium truncate max-w-[120px]">{item.feature}</span>
                      </p>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </motion.div>
        ) : (
          <div className="flex flex-col items-center justify-center py-10 text-center">
            <div className="w-16 h-16 bg-gray-100 dark:bg-gray-700 rounded-full flex items-center justify-center mb-4">
              <Clock className="h-8 w-8 text-gray-400 dark:text-gray-500" />
            </div>
            <h4 className="text-gray-700 dark:text-gray-300 font-medium mb-2">No History Yet</h4>
            <p className="text-gray-500 dark:text-gray-400 text-sm max-w-xs">
              Select a segment and click "Analyze" to start building your analysis history.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default ClusterHistorySection

