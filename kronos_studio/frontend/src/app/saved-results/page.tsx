"use client";

import { AppSidebar } from "./components/app-sidebar";
import { SidebarHeader } from "./components/SidebarHeader";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";

import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { deleteSavedResult, getSavedResult, listSavedResults } from "@/lib/api/savedResults";
import type { SavedResultMeta } from "@/schemas/resultsSchema";
import SavedResultViewer from "./components/SavedResultViewer";

export default function Page() {
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const queryClient = useQueryClient();

  const savedResultsQuery = useQuery({
    queryKey: ["saved-results"],
    queryFn: listSavedResults
  });

  const sortedResults = useMemo(
    () =>
      [...(savedResultsQuery.data ?? [])].sort(
        (left, right) => new Date(right.created_at).getTime() - new Date(left.created_at).getTime()
      ),
    [savedResultsQuery.data]
  );

  const activeId = selectedId ?? sortedResults[0]?.id ?? null;

  const detailQuery = useQuery({
    queryKey: ["saved-results", activeId],
    queryFn: () => getSavedResult(activeId as string),
    enabled: Boolean(activeId)
  });

  const deleteMutation = useMutation({
    mutationFn: deleteSavedResult,
    onSuccess: async (_, deletedId) => {
      queryClient.removeQueries({ queryKey: ["saved-results", deletedId] });
      queryClient.setQueryData<SavedResultMeta[]>(["saved-results"], (current) =>
        current?.filter((result) => result.id !== deletedId) ?? []
      );

      if (activeId === deletedId) {
        setSelectedId(null);
      }

      await queryClient.invalidateQueries({ queryKey: ["saved-results"] });
      toast.success("Saved result deleted");
    },
    onError: (error) => {
      toast.error("Failed to delete saved result", {
        description: error instanceof Error ? error.message : "Please try again."
      });
    }
  });

  return (
    <SidebarProvider
      style={
        {
          "--sidebar-width": "calc(var(--spacing) * 72)",
          "--header-height": "calc(var(--spacing) * 12)"
        } as React.CSSProperties
      }
    >
      <AppSidebar
        savedResults={{
          results: sortedResults,
          selectedId: activeId,
          isLoading: savedResultsQuery.isLoading,
          isError: savedResultsQuery.isError,
          deletingId: deleteMutation.isPending ? deleteMutation.variables ?? null : null,
          onSelect: (id) => {
            setSelectedId(id);
          },
          onRetry: () => {
            savedResultsQuery.refetch();
          },
          onDelete: (id) => {
            deleteMutation.mutate(id);
          }
        }}
      />
      <SidebarInset>
        <SidebarHeader savedResultsCount={sortedResults.length} />
        <div className='flex flex-1 flex-col'>
          <div className='@container/main flex flex-1 flex-col gap-2'>
            <div className='flex flex-col gap-4 py-4 md:gap-6 md:py-6'>
              <SavedResultViewer
                detail={detailQuery.data}
                isLoading={detailQuery.isLoading || detailQuery.isFetching}
                isError={detailQuery.isError}
                onRetry={() => {
                  detailQuery.refetch();
                }}
              />

              <div className='px-4 lg:px-6'>{/* <ChartAreaInteractive /> */}</div>
              {/* <DataTable data={data} /> */}
            </div>
          </div>
        </div>
      </SidebarInset>
    </SidebarProvider>
  );
}
