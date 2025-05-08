"use client"
import { motion } from "framer-motion"

const Loader = ({ size = "medium", text = "Loading..." }) => {
  const sizeClasses = {
    small: "h-8 w-8 border-2",
    medium: "h-12 w-12 border-3",
    large: "h-16 w-16 border-4",
  }

  return (
    <div className="flex flex-col items-center justify-center p-8">
      <motion.div
        className={`spinner ${sizeClasses[size]} border-indigo-200 border-t-indigo-600 rounded-full`}
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Number.POSITIVE_INFINITY, ease: "linear" }}
      />
      {text && (
        <motion.p
          className="mt-4 text-sm text-gray-600 dark:text-gray-400 font-medium"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          {text}
        </motion.p>
      )}
    </div>
  )
}

export default Loader

