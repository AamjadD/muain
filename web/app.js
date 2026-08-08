$(document).ready(function () {
  const translations = {
    ar: {
      documentTitle: "معين",
      languageLabel: "اللغة",
      title: "معين",
      subtitle: "تصنيف القضايا وربطها بالسند الشرعي",
      inputLabel: "أدخل نص القضية",
      analyze: "تحليل القضية",
      loading: "جارٍ التحليل...",
      resultHeading: "النتيجة",
      predictedClassLabel: "التصنيف المتوقع:",
      legalReferenceLabel: "السند الشرعي:",
      similarityScoreLabel: "درجة التشابه:",
      errorEmpty: "يرجى إدخال نص القضية أولاً.",
      errorGeneric: "حدث خطأ غير متوقع."
    }
  };

  // Bilingual support is intentionally disabled for the Arabic-only UI.
  // const savedLang = window.localStorage.getItem("demoLang");
  // const browserPrefersArabic = (navigator.language || "").toLowerCase().startsWith("ar");
  // let currentLang = savedLang === "ar" || savedLang === "en"
  //   ? savedLang
  //   : (browserPrefersArabic ? "ar" : "en");
  let currentLang = "ar";

  function translate(key) {
    return translations.ar[key] || "";
  }

  function applyTranslations() {
    const isArabic = currentLang === "ar";

    document.title = translate("documentTitle");
    $("html").attr("lang", currentLang).attr("dir", isArabic ? "rtl" : "ltr");
    $("body").attr("data-lang", currentLang);
    $("#caseText").attr("lang", currentLang).attr("dir", isArabic ? "rtl" : "ltr");

    $("[data-i18n]").each(function () {
      const key = $(this).data("i18n");
      $(this).text(translate(key));
    });

    $("[data-i18n-placeholder]").each(function () {
      const key = $(this).data("i18n-placeholder");
      $(this).attr("placeholder", translate(key));
    });

    // Language switching is disabled while the UI is Arabic-only.
    // $(".lang-btn").each(function () {
    //   const isActive = $(this).data("lang") === currentLang;
    //   $(this).toggleClass("active", isActive).attr("aria-pressed", isActive ? "true" : "false");
    // });
  }

  // function setLanguage(lang) {
  //   currentLang = lang === "ar" ? "ar" : "en";
  //   window.localStorage.setItem("demoLang", currentLang);
  //   applyTranslations();
  // }

  function pickValue(response, snakeKey, camelKey) {
    if (response[camelKey] !== undefined && response[camelKey] !== null) {
      return response[camelKey];
    }
    return response[snakeKey];
  }

  applyTranslations();

  // $(".lang-btn").on("click", function () {
  //   setLanguage($(this).data("lang"));
  // });

  $("#analyzeBtn").on("click", function () {
    const caseText = $("#caseText").val().trim();

    $("#errorBox").addClass("hidden").text("");
    $("#resultCard").addClass("hidden");

    if (!caseText) {
      $("#errorBox").removeClass("hidden").text(translate("errorEmpty"));
      return;
    }

    $("#loading").removeClass("hidden");

    $.ajax({
      url: "/predict",
      method: "POST",
      contentType: "application/json",
      data: JSON.stringify({ case_text: caseText, lang: currentLang }),
      success: function (response) {
        $("#loading").addClass("hidden");

        const predictedClass = pickValue(response, "predicted_class", "predictedClass") || "";
        const legalReference = pickValue(response, "legal_reference", "legalReference") || "";
        const similarityScore = pickValue(response, "similarity_score", "similarityScore");
        // const matchedCaseText = pickValue(response, "matched_case_text", "matchedCaseText") || "";

        $("#predictedClass").text(predictedClass);
        $("#legalReference").text(legalReference);
        $("#similarityScore").text(
          similarityScore !== undefined && similarityScore !== null ? Number(similarityScore).toFixed(4) : ""
        );
        // $("#matchedCaseText").text(matchedCaseText);

        $("#resultCard").removeClass("hidden");
      },
      error: function (xhr) {
        $("#loading").addClass("hidden");

        let message = translate("errorGeneric");
        if (xhr.responseJSON && xhr.responseJSON.error) {
          message = xhr.responseJSON.error;
        }

        $("#errorBox").removeClass("hidden").text(message);
      }
    });
  });
});
