"use client"
import { motion } from "framer-motion"
import { TrendingUp, TrendingDown, BarChart2 } from "lucide-react"

const ClusterDropdown = ({ groupedClusters, feature, clusterIndex, handleCellClick }) => {
  // Animation variants
  const dropdownVariants = {
    hidden: { opacity: 0, y: -5 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.2 } },
  }

  const itemVariants = {
    hidden: { opacity: 0, x: -5 },
    visible: { opacity: 1, x: 0 },
  }

  return (
    <motion.div
      variants={dropdownVariants}
      initial="hidden"
      animate="visible"
      exit="hidden"
      className="z-10 mx-auto mt-1 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 shadow-lg rounded-md p-2"
    >
      <motion.div className="space-y-2" variants={itemVariants}>
        <div
          className="cursor-pointer p-2 rounded-md hover:bg-indigo-50 dark:hover:bg-indigo-900/20 transition-colors flex items-center gap-2"
          onClick={() =>
            handleCellClick(
              feature,
              clusterIndex,
              groupedClusters.top2?.[feature]?.[clusterIndex]?.original?.Value ??
                groupedClusters.mean?.[feature]?.[clusterIndex]?.original?.Mean ??
                0,
            )
          }
        >
          <TrendingUp className="h-4 w-4 text-green-500 dark:text-green-400 flex-shrink-0" />
          <div className="flex-1">
            <div className="flex justify-between items-center">
              <span className="font-medium text-gray-800 dark:text-gray-200">Top 2 Value:</span>
              <span className="text-gray-600 dark:text-gray-400">
                {groupedClusters.top2[feature]?.[clusterIndex]?.original.Value || "N/A"}
              </span>
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              {groupedClusters.top2[feature]?.[clusterIndex]?.original.Percentage || "N/A"}%
            </div>
          </div>
        </div>

        <div
          className="cursor-pointer p-2 rounded-md hover:bg-indigo-50 dark:hover:bg-indigo-900/20 transition-colors flex items-center gap-2"
          onClick={() =>
            handleCellClick(
              feature,
              clusterIndex,
              groupedClusters.top3?.[feature]?.[clusterIndex]?.original?.Value ??
                groupedClusters.mean?.[feature]?.[clusterIndex]?.original?.Mean ??
                0,
            )
          }
        >
          <BarChart2 className="h-4 w-4 text-blue-500 dark:text-blue-400 flex-shrink-0" />
          <div className="flex-1">
            <div className="flex justify-between items-center">
              <span className="font-medium text-gray-800 dark:text-gray-200">Top 3 Value:</span>
              <span className="text-gray-600 dark:text-gray-400">
                {groupedClusters.top3[feature]?.[clusterIndex]?.original.Value || "N/A"}
              </span>
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              {groupedClusters.top3[feature]?.[clusterIndex]?.original.Percentage || "N/A"}%
            </div>
          </div>
        </div>

        <div
          className="cursor-pointer p-2 rounded-md hover:bg-indigo-50 dark:hover:bg-indigo-900/20 transition-colors flex items-center gap-2"
          onClick={() =>
            handleCellClick(
              feature,
              clusterIndex,
              groupedClusters.lowest[feature]?.[clusterIndex]?.original.Value ??
                groupedClusters.mean?.[feature]?.[clusterIndex]?.original?.Mean ??
                0,
            )
          }
        >
          <TrendingDown className="h-4 w-4 text-red-500 dark:text-red-400 flex-shrink-0" />
          <div className="flex-1">
            <div className="flex justify-between items-center">
              <span className="font-medium text-gray-800 dark:text-gray-200">Least Value:</span>
              <span className="text-gray-600 dark:text-gray-400">
                {groupedClusters.lowest[feature]?.[clusterIndex]?.original.Value || "N/A"}
              </span>
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-500">
              {groupedClusters.lowest[feature]?.[clusterIndex]?.original.Percentage || "N/A"}%
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}

export default ClusterDropdown

