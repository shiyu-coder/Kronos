"use client";

import { useMemo, useState } from "react";
import Link from "next/link";
import { ArrowLeft01Icon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { SidebarInset, SidebarProvider } from "@/components/ui/sidebar";
import SavedResultsSidebar from "@/components/saved-results/SavedResultsSidebar";
import SavedResultViewer from "@/components/saved-results/SavedResultViewer";
import { deleteSavedResult, getSavedResult, listSavedResults } from "@/lib/api/savedResults";
import type { SavedResultMeta } from "@/schemas/resultsSchema";

export default function SavedResultsPage() {
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
    <SidebarProvider className="flex-col md:flex-row">
      <SavedResultsSidebar
        results={sortedResults}
        selectedId={activeId}
        isLoading={savedResultsQuery.isLoading}
        isError={savedResultsQuery.isError}
        deletingId={deleteMutation.isPending ? deleteMutation.variables ?? null : null}
        onSelect={(id) => {
          setSelectedId(id);
        }}
        onRetry={() => {
          savedResultsQuery.refetch();
        }}
        onDelete={(id) => {
          deleteMutation.mutate(id);
        }}
      />

      <SidebarInset>
        <div className="sticky top-0 z-30 flex items-center justify-between border-b bg-background/90 px-4 py-3 backdrop-blur sm:px-6">
          <Button asChild variant="outline" size="sm">
            <Link href="/">
              <HugeiconsIcon icon={ArrowLeft01Icon} />
              Studio
            </Link>
          </Button>
          <p className="text-xs text-muted-foreground">{sortedResults.length} saved results</p>
        </div>

        <SavedResultViewer
          detail={detailQuery.data}
          isLoading={detailQuery.isLoading || detailQuery.isFetching}
          isError={detailQuery.isError}
          onRetry={() => {
            detailQuery.refetch();
          }}
        />
      </SidebarInset>
    </SidebarProvider>
  );
}
