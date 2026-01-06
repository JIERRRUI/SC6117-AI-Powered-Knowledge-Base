import React from 'react';
import { XIcon, NetworkIcon } from './Icons'; 

interface ConfirmationModalProps {
  isOpen: boolean;
  title: string;
  message: React.ReactNode;
  onConfirm: () => void;
  onCancel: () => void;
}

const ConfirmationModal: React.FC<ConfirmationModalProps> = ({ 
  isOpen, title, message, onConfirm, onCancel 
}) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#1e1e1e] border border-gray-700 rounded-xl shadow-2xl w-full max-w-md overflow-hidden transform scale-100 transition-all">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-700 flex justify-between items-center bg-gray-800/50">
          <h3 className="text-lg font-semibold text-white flex items-center gap-2">
            <NetworkIcon className="w-5 h-5 text-blue-400" />
            {title}
          </h3>
          <button onClick={onCancel} className="text-gray-400 hover:text-white transition-colors">
            <XIcon className="w-5 h-5" />
          </button>
        </div>

        {/* Body */}
        <div className="px-6 py-6 text-gray-300">
          {message}
        </div>

        {/* Footer */}
        <div className="px-6 py-4 bg-gray-800/30 flex justify-end gap-3 border-t border-gray-700">
          <button 
            onClick={onCancel}
            className="px-4 py-2 rounded-lg text-sm font-medium text-gray-400 hover:text-white hover:bg-white/10 transition-all"
          >
            Cancel
          </button>
          <button 
            onClick={onConfirm}
            className="px-4 py-2 rounded-lg text-sm font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-lg shadow-blue-900/20 transition-all"
          >
            Confirm Move
          </button>
        </div>
      </div>
    </div>
  );
};

export default ConfirmationModal;