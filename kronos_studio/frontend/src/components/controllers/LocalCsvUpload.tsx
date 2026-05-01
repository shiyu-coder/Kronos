"use client";

import { useRef, type ChangeEvent } from "react";
import { Upload, X } from "lucide-react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { uploadLocalDataFile } from "@/lib/api/prediction";
import { cn } from "@/lib/utils";
import { PredictionApiError } from "@/utils/handleAPI";

type LocalCsvUploadProps = {
  storedPath?: string | null;
  filename?: string | null;
  className?: string;
  compact?: boolean;
  onUploaded: (payload: { storedPath: string; filename: string }) => void;
  onClear: () => void;
};

export default function LocalCsvUpload({
  storedPath,
  filename,
  className,
  compact = false,
  onUploaded,
  onClear
}: LocalCsvUploadProps) {
  const inputRef = useRef<HTMLInputElement>(null);

  const uploadMutation = useMutation({
    mutationFn: uploadLocalDataFile,
    onSuccess: (data) => {
      onUploaded({ storedPath: data.stored_path, filename: data.filename });
      toast.success("CSV uploaded", {
        description: `${data.filename} (${data.row_count} rows)`
      });
    },
    onError: (error) => {
      const apiErr = error instanceof PredictionApiError ? error : null;
      toast.error(apiErr?.code ?? "Upload failed", {
        description: apiErr?.message ?? error.message ?? "Could not upload the CSV file."
      });
    }
  });

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    event.target.value = "";
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".csv")) {
      toast.error("Unsupported file", { description: "Please choose a CSV file." });
      return;
    }

    uploadMutation.mutate(file);
  };

  const displayName = filename || storedPath?.split(/[\\/]/).pop();

  return (
    <div className={cn("grid gap-1", className)}>
      <input ref={inputRef} type='file' accept='.csv,text/csv' className='hidden' onChange={handleFileChange} />
      <div className='flex min-w-0 gap-2'>
        <Button
          type='button'
          variant='outline'
          size={compact ? "sm" : "default"}
          className='flex-1 cursor-pointer'
          disabled={uploadMutation.isPending}
          onClick={() => inputRef.current?.click()}
        >
          <Upload size={compact ? 12 : 14} />
          {uploadMutation.isPending ? "Uploading..." : displayName ? "Replace CSV" : "Upload CSV"}
        </Button>
        {storedPath && (
          <Button
            type='button'
            variant='ghost'
            size={compact ? "icon-sm" : "icon"}
            className='cursor-pointer text-muted-foreground'
            onClick={onClear}
            disabled={uploadMutation.isPending}
            aria-label='Clear uploaded CSV'
          >
            <X size={compact ? 12 : 14} />
          </Button>
        )}
      </div>
      {displayName && (
        <p className='truncate text-xs text-muted-foreground' title={displayName}>
          {displayName}
        </p>
      )}
    </div>
  );
}
